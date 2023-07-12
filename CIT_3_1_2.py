# tomography_v2.py
# tomography

# 2023/02/17 ver0.0 : Start
# 2023/02/28 ver0.1 : Primary Version. (for 20 obs,12x12x11 grid -> 0.6sec. for 20 obs,22x22x21 grid -> 9.4sec)
# 2023/04/11 ver1.0 : New Version

# ! CAUTION !
# This program only support below format input data
#
# Put all input files following this guide
# [Anyfolder]/[Country code(ex:jp)]/[Year (4 digit)(ex:2023)]/[Day (3 digit)(ex:001)]/[Satelite Code (1 Alphabet + 2 digit)(ex:G01)]/[Receiver Code (4 digit)(ex:0001)].dat
#
# Input File Format follow :
# [Time (UT)] [STEC or Error (TECU)] [Satelite ECEF X (km)] [Satelite ECEF Y] [Satelite ECEF Z] [Receiver ECEF X] [Receiver ECEF Y] [Receiver ECEF Z] [Satelite ID] [Receiver ID]

import datetime
import glob
import math
import os
import time

import iri2016
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy.interpolate import interp1d
from scipy.optimize import lsq_linear, nnls
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import inv, spsolve

# # Settings # # # # # # # # # # # # #
mother_folder = "D:/unbias"  # place of teq file
IRI_folder = "D:/IRI2016"  # place of IRI file
Output_folder = "D:/tomo"
# # # # # # # # # # # # # # # # # # #


# CONSTANT

rf = 1.0/298.257223563
ra = 6378.1370
rb = ra*(1.0-rf)
re = math.sqrt((ra*ra-rb*rb)/(ra*ra))

# BASIC CLASS


class BLH:
    # b,l...[degree]
    # h...[km]
    b: float = 0.0
    l: float = 0.0
    h: float = 0.0

    def __init__(self, b, l, h):
        self.b = b
        self.l = l
        self.h = h

    def to_XYZ(self):
        answer = XYZ(0.0, 0.0, 0.0)
        n = ra/math.sqrt(1.0-re*re*math.sin(math.radians(self.b))
                         * math.sin(math.radians(self.b)))
        answer.x = (n+self.h)*math.cos(math.radians(self.b)) * \
            math.cos(math.radians(self.l))
        answer.y = (n+self.h)*math.cos(math.radians(self.b)) * \
            math.sin(math.radians(self.l))
        answer.z = ((1-re*re)*n+self.h)*math.sin(math.radians(self.b))
        return answer

    def __str__(self):
        return "[ B: "+str(self.b)+" L: "+str(self.l)+" H: "+str(self.h)+" ]"


class XYZ:

    # x,y,z...[km]
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def to_BLH(self):
        X = float(self.x)
        Y = float(self.y)
        Z = float(self.z)
        answer = BLH(0.0, 0.0, 0.0)
        # 1
        p = math.sqrt(X*X+Y*Y)
        h = ra*ra-rb*rb
        t = math.atan2(Z*ra, p*rb)  # rad
        answer.l = math.degrees(math.atan2(Y, X))  # deg
        # 2
        answer.b = math.degrees(math.atan2(
            (ra*rb*Z+ra*h*math.sin(t)**3), (ra*rb*p-rb*h*math.cos(t)**3)))  # deg
        # 3
        n = ra/math.sqrt(1-re*re*math.sin(math.radians(answer.b))
                         * math.sin(math.radians(answer.b)))
        # 4
        answer.h = p/math.cos(math.radians(answer.b))-n
        return answer

    def __str__(self):
        return "[ X: "+str(self.x)+" Y: "+str(self.y)+" Z: "+str(self.z)+" ]"

    def __add__(self, other):
        spo = XYZ(self.x+other.x, self.y+other.y, self.z+other.z)
        return spo

    def __sub__(self, other):
        smo = XYZ(self.x-other.x, self.y-other.y, self.z-other.z)
        return smo

    def __mul__(self, other):
        sto = XYZ(self.x*other, self.y*other, self.z*other)
        return sto

    def __rmul__(self, other):
        sto = XYZ(self.x*other, self.y*other, self.z*other)
        return sto

    def L2(self) -> float:
        siz = self.x**2+self.y**2+self.z**2
        return math.sqrt(siz)

# Subroutine
# # Generate Tomograpy Circumstance # #


def GenerateGrid():
    # Number of Grid for each axis (b=lat,l=lon,h=hgt)
    global z_b, z_l, z_h
    z_b = 29
    z_l = 29
    z_h = 27
    # Number of Marin Grid
    global mgn, hgt_mgn
    mgn = 3
    hgt_mgn = 0
    # zone + margin
    global n_b, n_l, n_h
    n_b = 48
    n_l = 48
    n_h = 26  # Only upper
    global n_all
    n_all = n_b*n_l*n_h

    # a1[?] ... Boundary positions of axis ?
    # a2[?] ... Center positions of axis ?
    global a1b, a1l, a1h
    a2b = np.full((n_b), 0.0, dtype=float)
    a2l = np.full((n_l), 0.0, dtype=float)
    a2h = np.full((n_h), 0.0, dtype=float)

    a1b, a1l, a1h = SetUpGrid()

    for ib in range(n_b):
        a2b[ib] = 0.5*(a1b[ib+1]+a1b[ib])
    for il in range(n_l):
        a2l[il] = 0.5*(a1l[il+1]+a1l[il])
    for ih in range(n_h):
        a2h[ih] = 0.5*(a1h[ih+1]+a1h[ih])

    return a1b, a1l, a1h, a2b, a2l, a2h


def SetUpGrid():
    # Size of boundary (Boundary of ?)
    global bob, bol, boh
    bob = 10.0
    bol = 10.0
    boh = 250.0

    global m_B, M_B, m_L, M_L, m_H, M_H
    m_B = -5.0
    M_B = 80.0
    m_L = 95.0
    M_L = 180.0
    m_H = 70.0
    M_H = 1010.0

    # a1[?] ... Boundary positions of axis ?
    a1b = np.full((n_b+1), 0.0, dtype=float)
    a1l = np.full((n_l+1), 0.0, dtype=float)
    a1h = np.full((n_h+1), 0.0, dtype=float)

    # a1b = np.array([-5.0, 5.0, 15.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 30.5,
    #                 31.0, 31.5, 32.0, 32.5, 33.0, 33.5, 34.0, 34.5, 35.0, 35.5,
    #                 36.0, 36.5, 37.0, 37.5, 38.0, 38.5, 39.0, 39.5, 40.0, 40.5,
    #                 41.0, 41.5, 42.0, 42.5, 43.0, 43.5, 44.0, 44.5, 45.0, 46.0,
    #                 47.0, 48.0, 49.0, 50.0, 60.0, 70.0, 80.0])
    a1b = np.array([-5.0, 05.0, 15.0, 20.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                    30.5, 31.0, 31.5, 32.0, 32.5, 33.0, 33.5, 34.0, 34.5, 35.0,
                    35.5, 36.0, 36.5, 37.0, 37.5, 38.0, 38.5, 39.0, 39.5, 40.0,
                    40.5, 41.0, 41.5, 42.0, 42.5, 43.0, 43.5, 44.0, 44.5, 45.0,
                    46.0, 47.0, 48.0, 49.0, 50.0, 55.0, 60.0, 70.0, 80.0])
    # a1b = np.array([-5.0, 05.0, 10.0, 15.0, 20.0, 23.0, 25.0, 26.0, 27.0,
    #                 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0,
    #                 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0,
    #                 48.0, 49.0, 50.0, 52.0, 55.0, 60.0, 65.0, 70.0, 80.0])
    # a1l = np.array([95.0, 105.0, 115.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 130.5,
    #                 131.0, 131.5, 132.0, 132.5, 133.0, 133.5, 134.0, 134.5, 135.0, 135.5,
    #                 136.0, 136.5, 137.0, 137.5, 138.0, 138.5, 139.0, 139.5, 140.0, 140.5,
    #                 141.0, 141.5, 142.0, 142.5, 143.0, 143.5, 144.0, 144.5, 145.0, 146.0,
    #                 147.0, 148.0, 149.0, 150.0, 160.0, 170.0, 180.0])
    a1l = np.array([095.0, 105.0, 115.0, 120.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0,
                    130.5, 131.0, 131.5, 132.0, 132.5, 133.0, 133.5, 134.0, 134.5, 135.0,
                    135.5, 136.0, 136.5, 137.0, 137.5, 138.0, 138.5, 139.0, 139.5, 140.0,
                    140.5, 141.0, 141.5, 142.0, 142.5, 143.0, 143.5, 144.0, 144.5, 145.0,
                    146.0, 147.0, 148.0, 149.0, 150.0, 155.0, 160.0, 170.0, 180.0])
    # a1l = np.array([095.0, 105.0, 110.0, 115.0, 120.0, 125.0, 126.0, 127.0, 128.0, 129.0,
    #                 130.0, 131.0, 132.0, 133.0, 134.0, 135.0, 136.0, 137.0, 138.0, 139.0,
    #                 140.0, 141.0, 142.0, 143.0, 144.0, 145.0, 146.0, 147.0, 148.0, 149.0,
    #                 150.0, 155.0, 160.0, 165.0, 170.0, 180.0])
    # a1h = np.array([75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 225.0, 250.0, 275.0,
    #                 300.0, 325.0, 350.0, 375.0, 400.0, 425.0, 450.0, 475.0, 500.0,
    #                 525.0, 550.0, 575.0, 600.0, 650.0, 700.0, 750.0, 800.0, 900.0, 1000.0])
    a1h = np.array([070.0, 100.0, 130.0, 160.0, 190.0, 220.0, 250.0, 280.0, 310.0, 340.0,
                    370.0, 400.0, 430.0, 460.0, 490.0, 520.0, 550.0, 580.0, 610.0, 660.0,
                    710.0, 760.0, 810.0, 860.0, 910.0, 960.0, 1010.0])
    return a1b, a1l, a1h

# # # # # # # # # # # # # # # # # # # #

# # Import observation File # # # # # # gfgfgfg


def ImportDataFromTomoI(country, year4, day, CODE, epoc: int):
    Tomography_input = "D:/tomoi/{c}/{y:04d}/{d:03d}/{code}/{ep:04d}.tomoi".format(
        c=country, y=year4, d=day, code="_______________", ep=epoc)
    data = []
    recs = []
    sats = []
    with open(Tomography_input, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            else:
                data.append(float(line.split()[0]))
                sat = XYZ(float(line.split()[1]), float(
                    line.split()[2]), float(line.split()[3]))
                sats.append(sat)
                rec = XYZ(float(line.split()[4]), float(
                    line.split()[5]), float(line.split()[6]))
                recs.append(rec)

    return np.array(data, dtype=float), sats, recs

# # # # # # # # # # # # # # # # # # # #

# # Calculation C # # # # # # # # # # #


def Generate_C(country,year4,day,code,lsat, lrec, a1b, a1l, a1h, ep, debug_mode=False) -> tuple[lil_matrix, np.ndarray]:
    os.makedirs(
        "D:/tomoc/{c}/{y:04d}/{d:03d}/{code}".format(c=country, y=year4, d=day, code=code), exist_ok=True)
    Tomography_C = "D:/tomoc/{c}/{y:04d}/{d:03d}/{code}/{epc:04d}.tomoc".format(
        c=country, y=year4, d=day, code=code, epc=ep)
    global n_obs
    n_obs = len(lsat)
    A = lil_matrix((n_obs, n_all))
    A_bool = np.full((n_all), False, dtype=bool)
    with open(Tomography_C, "w") as f:
        print(n_obs, n_all, file=f)
        for iobs in range(n_obs):
            A_idx, A_data = Generate_C_row(
                lsat[iobs], lrec[iobs], a1b, a1l, a1h, debug_mode)
            for jidx in range(len(A_idx)):
                A[iobs, A_idx[jidx]] = A_data[jidx]
                A_bool[A_idx[jidx]] = True
                print(iobs, A_idx[jidx], A_data[jidx], file=f)
    return A, A_bool


# return IRI value at hgt[km] -> TECU/km

def inner(x1: XYZ, x2: XYZ) -> float:
    ans = 0.0
    ans += x1.x*x2.x
    ans += x1.y*x2.y
    ans += x1.z*x2.z
    return ans

# return zenith theta[rad]


def zenith(rec: XYZ, sat: XYZ, H: float):
    t, ipp = spH(rec, sat, H)
    cosZ = inner(ipp-rec, ipp)/math.sqrt(inner(ipp, ipp)
                                         * inner(ipp-rec, ipp-rec))
    return math.acos(cosZ)


# H ... Reference Altitude [km]
# m_B-M_B , m_L-M_L ... Tomography Range
# nb,nl ... number of Reference points

def Import_iri2016(day, H, m_B, M_B, m_L, M_L, nb, nl):
    _b = np.linspace(m_B, M_B, nb)
    _l = np.linspace(m_L, M_L, nl)
    bs, ls = np.meshgrid(_b, _l)

    ans = np.full((nb, nl), 0.0, dtype=float)

    for ib in range(nb):
        for jl in range(nl):
            ans[ib, jl] = iri2016.IRI(
                day, [H, H, 1], bs[ib, jl], ls[ib, jl]).ne[0]*1.0e+3
    return ans


def _interpolate(ans, m_B, M_B, m_L, M_L, nb, nl, lat, lon):
    db = (M_B-m_B)/(nb-1)
    dl = (M_L-m_L)/(nl-1)
    i_b = math.floor((lat-m_B)/db)
    i_l = math.floor((lon-m_L)/dl)
    bef_b = m_B+db*i_b
    aft_b = m_B+db*(i_b+1)
    bef_l = m_L+dl*i_l
    aft_l = m_L+dl*(i_l+1)
    Ne = 0.0
    Ne += ((lat-bef_b)/db)*((lon-bef_l)/dl)*ans[i_b, i_l]
    Ne += ((lat-bef_b)/db)*((aft_l-lon)/dl)*ans[i_b, i_l+1]
    Ne += ((aft_b-lat)/db)*((lon-bef_l)/dl)*ans[i_b+1, i_l]
    Ne += ((aft_b-lat)/db)*((aft_l-lon)/dl)*ans[i_b+1, i_l+1]
    return Ne


def Del_Plasmaphic(tecs, recs, sats, H, year4, doy, epoch):
    UT = epoch/120.0
    tmp = datetime.datetime(year=year4, month=1, day=1)+datetime.timedelta(
        days=doy-1, hours=UT
    )
    year = tmp.year
    month = tmp.month
    day = tmp.day
    hour = tmp.hour
    minute = tmp.minute
    second = tmp.second
    day = "{y:04d}-{m:02d}-{d:02d} {h:02d}:{mn:02d}:{s:02d}".format(
        y=year, m=month, d=day, h=hour, mn=minute, s=second
    )
    nb = 7
    nl = 7
    _dIRI = Import_iri2016(day, H, m_B, M_B, m_L, M_L, nb, nl)

    tecs_copy = tecs.copy()
    Lobs = len(tecs)
    scale = 1.0e+3
    for iobs in range(Lobs):
        theta = zenith(recs[iobs], sats[iobs], H)
        t, IPP = spH(recs[iobs], sats[iobs], H)
        dtec = _interpolate(_dIRI, m_B, M_B, m_L, M_L, nb,
                            nl, IPP.to_BLH().b, IPP.to_BLH().l)*scale/1.0e+16*abs(math.cos(theta))
        tecs_copy[iobs] -= dtec
    return csr_matrix(tecs_copy)


# -1 / 0 / 1 / 2 / 3 / 4 / 5 / 6
#    / 0 / 1 / 3 / 5 / 7 / 11
# -3       2                   14
#

def lnearst_idx(lst: list, d: float):
    L = len(lst)
    if lst[0] > d:
        return -1
    for i in range(L):
        if lst[i] > d:
            return i-1
        if i == L-1:
            return L


def Generate_C_row(sat, rec, a1b, a1l, a1h, debug_mode=False):
    cp = {}
    C_idx = []
    C_data = []
    tsat, tmpsat = spH(rec, sat, a1h[n_h], False)
    for ib in range(n_b+1):
        tipp, ipp = spB(rec, tmpsat, a1b[ib], False)
        if abs(ipp.to_BLH().b-a1b[ib]) > 0.1:
            pass
        else:
            cp[tipp*tsat] = ipp
    for il in range(n_l+1):
        tipp, ipp = spL(rec, sat, a1l[il])
        if abs(ipp.to_BLH().l-a1l[il]) > 0.1:
            pass
        else:
            cp[tipp] = ipp
    for ih in range(n_h+1):
        tipp, ipp = spH(rec, sat, a1h[ih], False)
        if abs(ipp.to_BLH().h-a1h[ih]) > 1.0:
            pass
        else:
            cp[tipp] = ipp
    if debug_mode:
        for k, v in enumerate(cp):
            print(v, cp[v].to_BLH())
    cp_sort = sorted(cp.items())
    list_key = []
    for key in cp_sort:
        list_key.append(key[0])
    for i in range(len(list_key)-1):
        bkey = list_key[i]
        akey = list_key[i+1]
        if bkey > 1.005 or akey < -0.005:
            pass
        else:
            bpos: XYZ = cp[bkey]  # XYZ
            apos: XYZ = cp[akey]  # XYZ
            mpos = bpos*0.5+apos*0.5
            mpos_blh = mpos.to_BLH()
            bidx = lnearst_idx(a1b, mpos_blh.b)
            lidx = lnearst_idx(a1l, mpos_blh.l)
            hidx = lnearst_idx(a1h, mpos_blh.h)
            if -1 < bidx < n_b and -1 < lidx < n_l and -1 < hidx < n_h:
                idx = lidx+bidx*n_l+hidx*n_b*n_l
                C_idx.append(idx)
                C_data.append((bpos-apos).L2())
                if debug_mode:
                    print(idx, "({l},{b},{h}):".format(l=lidx, b=bidx, h=hidx), bpos.to_BLH(), "->",
                          apos.to_BLH(), ":", (bpos-apos).L2())
                    input()

    return C_idx, C_data  # , b_ipp, a_ipp


# [m]
# float return is distance parameter, normalize like
#  Receiver                       Satelite
#    0.0     -> -> -> -> -> ->     1.0

# returnのfloatはRECで0,SATで1となるように標準化した距離パラメータ

# at rec ... t = 0.0
# at sat ... t = 1.0

def spH(rec: XYZ, sat: XYZ, H, debug_mode=False, t=1.0) -> tuple[float, XYZ]:
    # 1[m]
    if debug_mode:
        print(1, rec.to_BLH(), sat.to_BLH())
    eps = 0.01
    dt = 0.01
    dt_inv = 100.0
    point = XYZ(rec.x*(1-t)+sat.x*t, rec.y*(1-t)+sat.y*t, rec.z*(1-t)+sat.z*t)
    if abs(point.to_BLH().h-H) < eps:
        if t < 0.0:
            return -1.0, rec
        elif t > 1.0:
            return 2.0, sat
        else:
            return t, point
    else:
        dpoint = XYZ(rec.x*(1-t-dt)+sat.x*(t+dt), rec.y*(1-t-dt) +
                     sat.y*(t+dt), rec.z*(1-t-dt)+sat.z*(t+dt))
        fbar_i = (dpoint.to_BLH().h-point.to_BLH().h)*dt_inv
        tbar = t-(point.to_BLH().h-H)/fbar_i
        return spH(rec, sat, H, debug_mode, tbar)


def spB(rec: XYZ, sat: XYZ, B, debug_mode=False, t=1.0) -> tuple[float, XYZ]:
    if debug_mode:
        print(2, t, B, (rec*(1-t)+sat*t).to_BLH())
    # 1.0e-5[deg]=1[m]
    eps = 1.0e-4
    dt = 0.00001
    point = XYZ(rec.x*(1-t)+sat.x*t, rec.y*(1-t)+sat.y*t, rec.z*(1-t)+sat.z*t)
    if t < 0.0 or t > 1.0:
        return -1.0, rec
    if abs(point.to_BLH().b-B) < eps:
        if t < 0.0:
            return -1.0, rec
        elif t > 1.0:
            return 2.0, sat
        else:
            return t, point
    else:
        dpoint = XYZ(rec.x*(1-t-dt)+sat.x*(t+dt), rec.y*(1-t-dt) +
                     sat.y*(t+dt), rec.z*(1-t-dt)+sat.z*(t+dt))
        fbar_i = (dpoint.to_BLH().b-point.to_BLH().b)/dt
        tbar = t-(point.to_BLH().b-B)/fbar_i
        return spB(rec, sat, B, debug_mode, tbar)


def spL(rec: XYZ, sat: XYZ, L) -> tuple[float, XYZ]:
    tanL = math.tan(math.radians(L))
    s = (sat.y-sat.x*tanL)/((rec.x-sat.x)*tanL-rec.y+sat.y)
    ans = XYZ(0.0, 0.0, 0.0)
    ans.x = s*rec.x+(1-s)*sat.x
    ans.y = s*rec.y+(1-s)*sat.y
    ans.z = s*rec.z+(1-s)*sat.z
    return 1.0-s, ans

# # # # # # # # # # # # # # # # # # # #

# # Generate 2nd term # # # # # # # # # # # # #

# H ... n_obs+n_all , n_all


def coeff(h):
    if h < 90.0:
        return 1.0e-1
    elif h < 200.0:
        return 2.0e-3
    elif h < 450.0:
        return 1.0e-3
    elif h < 600.0:
        return 3.0e-3
    elif h < 700.0:
        return 4.0e-3
    elif h < 800.0:
        return 1.0e-2
    elif h < 900.0:
        return 1.0e-1
    else:
        return 1.0e0


def Generate_H(a2b, a2l, a2h, alpha: float) -> csr_matrix:
    M = n_all
    H = lil_matrix((M, M))
    for ih in range(n_h):
        for jb in range(n_b):
            for kl in range(n_l):
                idx = kl+jb*n_l+ih*n_l*n_b
                coeff_sum = 0.0
                # west
                if kl != 0:
                    idx1 = kl-1+jb*n_l+ih*n_l*n_b
                    H[idx, idx1] = coeff(a2h[ih])*-1.0*alpha
                    coeff_sum -= H[idx, idx1]
                # east
                if kl != n_l-1:
                    idx2 = kl+1+jb*n_l+ih*n_l*n_b
                    H[idx, idx2] = coeff(a2h[ih])*-1.0*alpha
                    coeff_sum -= H[idx, idx2]
                # south
                if jb != 0:
                    idx3 = kl+(jb-1)*n_l+ih*n_l*n_b
                    H[idx, idx3] = coeff(a2h[ih])*-1.0*alpha
                    coeff_sum -= H[idx, idx3]
                # north
                if jb != n_b-1:
                    idx4 = kl+(jb+1)*n_l+ih*n_l*n_b
                    H[idx, idx4] = coeff(a2h[ih])*-1.0*alpha
                    coeff_sum -= H[idx, idx4]
                # below
                if ih != 0:
                    idx5 = kl+jb*n_l+(ih-1)*n_l*n_b
                    H[idx, idx5] = coeff(a2h[ih-1])*-1.0
                    coeff_sum -= H[idx, idx5]
                # above
                if ih < n_h-1:
                    idx6 = kl+jb*n_l+(ih+1)*n_l*n_b
                    H[idx, idx6] = coeff(a2h[ih+1])*-1.0
                    coeff_sum -= H[idx, idx6]
                if ih == 0 or ih == n_h-1:
                    H[idx, idx] = coeff_sum+coeff(a2h[ih])*1.0
                else:
                    H[idx, idx] = coeff_sum
    return H.tocsr()


def Generate_Y(H, a2b, a2l, year4, doy, epoch):
    Y = [0.0 for i in range(n_all)]
    UT = epoch/120.0
    tmp = datetime.datetime(year=year4, month=1, day=1)+datetime.timedelta(
        days=doy-1, hours=UT
    )
    year = tmp.year
    month = tmp.month
    day = tmp.day
    hour = tmp.hour
    minute = tmp.minute
    second = tmp.second
    day = "{y:04d}-{m:02d}-{d:02d} {h:02d}:{mn:02d}:{s:02d}".format(
        y=year, m=month, d=day, h=hour, mn=minute, s=second
    )
    nb = 7
    nl = 7
    _dIRI = Import_iri2016(day, H, m_B, M_B, m_L, M_L, nb, nl)
    for jb in range(n_b):
        for kl in range(n_l):
            idx1 = kl+jb*n_l+(n_h-1)*n_l*n_b
            Y[idx1] = _interpolate(_dIRI, m_B, M_B, m_L,
                                   M_L, nb, nl, a2b[jb], a2l[kl])/1.0e+16
    return csr_matrix(Y)


# # # # # # # # # # # # # # # # # # # #

# # Tomography # # # # # # # # # # # # #


def Tomography(A: csr_matrix, B: csr_matrix, H: csr_matrix, Y: csr_matrix, co: float):

    X = spsolve(A.T*A+co*co*H.T*H, A.T*B.T+co*co*H.T*Y.T)

    return X

# # # # # # # # # # # # # # # # # # # #

# # OUTPUT # # # # # # # # # # # # # #


def puts(B, country, year, day, CODE,  epoch):
    os.makedirs("D:/tomob/{c}/{y:04d}/{d:03d}/{code}".format(
        c=country, y=year, d=day, ep=epoch, code="_______________"
    ), exist_ok=True)
    tomob = "D:/tomob/{c}/{y:04d}/{d:03d}/{code}/{ep:04d}.tomob".format(
        c=country, y=year, d=day, ep=epoch, code="_______________"
    )
    with open(tomob, "w") as bf:
        for i in range(B.shape[1]):
            if B[0, i] > 1.0e-5:
                print(i, B[0, i], file=bf)


def OutPut(X, A_bool, op_file, epoch):
    # print(X.shape)
    ut = epoch/120.0
    dut = datetime.timedelta(hours=ut)
    utsec = dut.seconds
    uth = utsec//3600
    utm = (utsec-uth*3600)//60
    uts = utsec % 60
    with open(op_file, "w") as f:
        print("# Tomography Result", file=f)
        print("# ", file=f)
        print("# RUN BY {prog}".format(prog="tomography.py"), file=f)
        print("# ", file=f)
        print("# UTC : {dt}".format(dt=datetime.datetime.now()), file=f)
        print("# ", file=f)
        print("# Number of Boxel (at main area)", file=f)
        print("# Latitude : {nb:03d}".format(nb=z_b), file=f)
        print("# Longitude : {nl:03d}".format(nl=z_l), file=f)
        print("# Height : {nh:03d}".format(nh=z_h), file=f)
        print("# margin : {nm:03d}".format(nm=mgn), file=f)
        print("# ", file=f)
        print("# Number of Boxel (All area)", file=f)
        print("# Latitude : {nb:03d}".format(nb=n_b), file=f)
        print("# Longitude : {nl:03d}".format(nl=n_l), file=f)
        print("# Height : {nh:03d}".format(nh=n_h), file=f)
        print("# ", file=f)
        print("# *** Plain List ***", file=f)
        print("# Latitude", file=f)
        for ib in range(n_b+1):
            print("# {b:+06.2f}".format(b=a1b[ib]), file=f)
        print("# Longitude", file=f)
        for il in range(n_l+1):
            print("# {l:+06.2f}".format(l=a1l[il]), file=f)
        print("# Height", file=f)
        for ih in range(n_h+1):
            print("# {h:+07.2f}".format(h=a1h[ih]), file=f)
        print("# ", file=f)
        print("# Featured Time {y:04d} / {d:03d} / {h:02d} : {m:02d} : {s:02d}".format(
            y=year4, d=day, h=uth, m=utm, s=uts), file=f)
        print("# ", file=f)
        print("# END OF HEADER", file=f)
        print("", file=f)
        for ih in range(n_h):
            for jb in range(n_b):
                for kl in range(n_l):
                    if True:
                        f.write(
                            "{teq:+15.13f} ".format(teq=X[kl+jb*n_l+ih*n_l*n_b]))
                    # else:
                    #     f.write(
                    #         "{teq:+15.13f} ".format(teq=np.nan))
                print("", file=f)
            print("", file=f)

# # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # #
# PARAMETERS
# Lat. : 20.0 - 50.0
# Lon. : 120.0 - 156.0
# Hgt. : 100.0 - 800.0
# Bound Lat. : 10.0 - 60.0
# Bound Lon. : 110.0 - 166.0
# Bound Hgt. : 100.0 - 1300.0

# # # # # # # # # # # # # # # # # # # #


# Main

def __Tomography__(c: str, y: int, d: int, CODE: str, ep: int, coeff: float, alpha: float):
    global year4, day, country, code
    year4 = y
    day = d
    country = c
    code = CODE

    print(
        "*** START TOMOGRAPHY : {y:04d} {d:03d} {epc:04d} : {t:09.3f} ***".format(y=year4, d=day, epc=ep, t=time.time()-start))
    # Set up grid
    a1b, a1l, a1h, a2b, a2l, a2h = GenerateGrid()

    # Import data file (default: .tomoi)
    tec, sat, rec = ImportDataFromTomoI(country, year4, day, CODE, epoc=ep)

    # Set up coefficient matrix C
    lil_A, A_bool = Generate_C(
        sat, rec, a1b, a1l, a1h, ep, debug_mode=False)

    # Delete effect of plasmapheric plasma
    B = Del_Plasmaphic(tec, rec, sat, H=M_H, year4=y, doy=d, epoch=ep)

    # Record right hand side of formula to file
    puts(B, c, y, d, CODE, ep)

    # Set up constraint matrix H
    csr_H = Generate_H(a2b, a2l, a2h, alpha)

    # Set up boundary vector Y
    csr_Y = Generate_Y(M_H, a2b, a2l, year4, d, ep)

    # Calculate hyper parameter scale
    csr_A = lil_A.tocsr()
    co = csr_matrix.trace(csr_A.T*csr_A)/csr_matrix.trace(csr_H.T*csr_H)

    print("Finish preparation for tomography {y:04d} {d:03d} {epc:04d} : {t:09.3f}".format(
        y=year4, d=day, epc=ep, t=time.time()-start))

    # Do tomography
    X = Tomography(csr_A, B, csr_H, csr_Y, coeff*co)

    print("Finish Tomography {y:04d} {d:03d} {epc:04d} : {t:09.3f}".format(
        y=year4, d=day, epc=ep, t=time.time()-start))

    return X, A_bool

        
    

start = time.time()

if __name__ == "__main__":
    country = "jp"
    year4 = 2016
    day = 193
    CODE = "tid_5+0_1d_h2__"
    epoch = 1320
    X, M_bool = __Tomography__(country, year4, day, CODE, epoch, 10.0, 1.0)

    Output_folder = "D:/tomo"

    os.makedirs(Output_folder+"/{c}/{y:04}/{d:03}/{code}".format(
        c=country, y=year4, d=day, code=CODE), exist_ok=True)
    op_file = Output_folder + \
        "/{c}/{y:04}/{d:03}/{code}/{epc:04d}.tomo".format(
            c=country, y=year4, d=day, code=CODE, epc=epoch)
    OutPut(X, M_bool, op_file, epoch)
    # print("End export.")
