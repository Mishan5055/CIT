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
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.optimize import lsq_linear, nnls
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import inv, spsolve, bicgstab
from discordwebhook import Discord

from concurrent.futures import ProcessPoolExecutor, as_completed

# # Settings # # # # # # # # # # # # #
drive = "D:"
# # # # # # # # # # # # # # # # # # #


# CONSTANT

rf = 1.0 / 298.257223563
ra = 6378.1370
rb = ra * (1.0 - rf)
re = math.sqrt((ra * ra - rb * rb) / (ra * ra))

# BASIC CLASS


def lnearst_idx(lst: list, d: float):
    L = len(lst)
    if lst[0] > d:
        return -1
    for i in range(L):
        if lst[i] > d:
            return i - 1
        if i == L - 1:
            return L - 1


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
        n = ra / math.sqrt(
            1.0
            - re * re * math.sin(math.radians(self.b)) * math.sin(math.radians(self.b))
        )
        answer.x = (
            (n + self.h)
            * math.cos(math.radians(self.b))
            * math.cos(math.radians(self.l))
        )
        answer.y = (
            (n + self.h)
            * math.cos(math.radians(self.b))
            * math.sin(math.radians(self.l))
        )
        answer.z = ((1 - re * re) * n + self.h) * math.sin(math.radians(self.b))
        return answer

    def __str__(self):
        return (
            "[ B: " + str(self.b) + " L: " + str(self.l) + " H: " + str(self.h) + " ]"
        )


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
        p = math.sqrt(X * X + Y * Y)
        h = ra * ra - rb * rb
        t = math.atan2(Z * ra, p * rb)  # rad
        answer.l = math.degrees(math.atan2(Y, X))  # deg
        # 2
        answer.b = math.degrees(
            math.atan2(
                (ra * rb * Z + ra * h * math.sin(t) ** 3),
                (ra * rb * p - rb * h * math.cos(t) ** 3),
            )
        )  # deg
        # 3
        n = ra / math.sqrt(
            1
            - re
            * re
            * math.sin(math.radians(answer.b))
            * math.sin(math.radians(answer.b))
        )
        # 4
        answer.h = p / math.cos(math.radians(answer.b)) - n
        return answer

    def __str__(self):
        return (
            "[ X: " + str(self.x) + " Y: " + str(self.y) + " Z: " + str(self.z) + " ]"
        )

    def __add__(self, other):
        spo = XYZ(self.x + other.x, self.y + other.y, self.z + other.z)
        return spo

    def __sub__(self, other):
        smo = XYZ(self.x - other.x, self.y - other.y, self.z - other.z)
        return smo

    def __mul__(self, other):
        sto = XYZ(self.x * other, self.y * other, self.z * other)
        return sto

    def __rmul__(self, other):
        sto = XYZ(self.x * other, self.y * other, self.z * other)
        return sto

    def L2(self) -> float:
        siz = self.x**2 + self.y**2 + self.z**2
        return math.sqrt(siz)


class EXPERIMENT:
    c: str = ""
    y: int = 0
    d: int = 0
    code: str = ""
    ep: int = 0

    def __init__(self, c, y, d, code, ep):
        self.c = c
        self.y = y
        self.d = d
        self.code = code
        self.ep = ep

    def __str__(self):
        return (
            self.c
            + " "
            + str(self.y)
            + " / "
            + str(self.d)
            + " "
            + self.code
            + " : "
            + str(self.ep)
        )


class SETTING:
    a1b: np.ndarray = np.array([])
    a1l: np.ndarray = np.array([])
    a1h: np.ndarray = np.array([])
    a2b: np.ndarray = np.array([])
    a2l: np.ndarray = np.array([])
    a2h: np.ndarray = np.array([])

    n_h: int = 0
    n_b: int = 0
    n_l: int = 0
    n_all: int = 0
    cof: float = 1.0e1
    alpha: float = 0.3

    m_H: float = 0.0
    M_H: float = 0.0
    m_B: float = 0.0
    M_B: float = 0.0
    m_L: float = 0.0
    M_L: float = 0.0

    H: csr_matrix = csr_matrix((1, 1))
    trH_2: float = 0.0

    def __init__(self, a1h, a1b, a1l, cof, alpha):
        self.a1h = a1h
        self.a1b = a1b
        self.a1l = a1l
        self.cof = cof
        self.alpha = alpha

        self.n_h = a1h.shape[0] - 1
        self.n_b = a1b.shape[0] - 1
        self.n_l = a1l.shape[0] - 1
        self.n_all = self.n_h * self.n_b * self.n_l

        self.a2h = np.full((self.n_h), 0.0, dtype=float)
        self.a2b = np.full((self.n_b), 0.0, dtype=float)
        self.a2l = np.full((self.n_l), 0.0, dtype=float)
        for ih in range(self.n_h):
            self.a2h[ih] = 0.5 * (self.a1h[ih] + self.a1h[ih + 1])
        for ib in range(self.n_b):
            self.a2b[ib] = 0.5 * (self.a1b[ib] + self.a1b[ib + 1])
        for il in range(self.n_l):
            self.a2l[il] = 0.5 * (self.a1l[il] + self.a1l[il + 1])

        self.m_H = np.min(self.a1h)
        self.M_H = np.max(self.a1h)
        self.m_B = np.min(self.a1b)
        self.M_B = np.max(self.a1b)
        self.m_L = np.min(self.a1l)
        self.M_L = np.max(self.a1l)

    def __initH__(self, H: csr_matrix):
        self.H = H
        self.trH_2 = csr_matrix.trace(self.H.T * self.H)

    def nbrock(self) -> tuple[int, int, int, int]:
        """_summary_
        settingのブロック数を返します。 \n
        return : n_all, n_h, n_b, n_l \n
        Returns: \n
            tuple[int, int, int, int]: _description_
        """
        return [self.n_all, self.n_h, self.n_b, self.n_l]

    def plains(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """_summary_
        settingに設定された境界を返します。\n
        return : a1h, a1b, a1l \n
        Returns: \n
            tuple[np.ndarray, np.ndarray, np.ndarray]: _description_
        """
        return [self.a1h, self.a1b, self.a1l]

    def centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return [self.a2h, self.a2b, self.a2l]

    def bounds(self) -> tuple[float, float, float, float, float, float]:
        return [self.m_H, self.M_H, self.m_B, self.M_B, self.m_L, self.M_L]

    def simprize(self, N_b, N_l):
        l2b = np.full((self.n_b), 0.0, dtype=float)
        l2l = np.full((self.n_l), 0.0, dtype=float)
        for ib in range(self.n_b):
            l2b[ib] = self.a1b[ib + 1] - self.a1b[ib]
        for il in range(self.n_l):
            l2l[il] = self.a1l[il + 1] - self.a1l[il]
        b2b = np.full((self.n_b + 1), True, dtype=bool)
        b2l = np.full((self.n_l + 1), True, dtype=bool)
        loop = 0
        while np.sum(b2b) > N_b:
            argMIN = -1
            MIN = 100.0
            lower = 0
            upper = 1
            while upper < self.n_b + 1:
                if b2b[upper]:
                    dif = self.a1b[upper] - self.a1b[lower]
                    if dif < MIN:
                        MIN = dif
                        argMIN = upper
                    lower = upper
                upper += 1
            b2b[argMIN] = False
            # print(loop, end=": ")
            # for ib in range(self.n_b+1):
            #     if b2b[ib]:
            #         print(self.a1b[ib], end=" ")
            # print("")
            loop += 1
        loop = 0
        while np.sum(b2l) > N_l:
            argMIN = -1
            MIN = 100.0
            lower = 0
            upper = 1
            while upper < self.n_l + 1:
                if b2l[upper]:
                    dif = self.a1l[upper] - self.a1l[lower]
                    if dif < MIN:
                        MIN = dif
                        argMIN = upper
                    lower = upper
                upper += 1
            b2l[argMIN] = False
            # print(loop, end=": ")
            # for il in range(self.n_l+1):
            #     if b2l[il]:
            #         print(self.a1l[il], end=" ")
            # print("")
            loop += 1
        simple_a1b = []
        simple_a1l = []
        for ib in range(self.n_b + 1):
            if b2b[ib]:
                simple_a1b.append(self.a1b[ib])
        for il in range(self.n_l + 1):
            if b2l[il]:
                simple_a1l.append(self.a1l[il])

        simprize_prob = SETTING(
            self.a1h, np.array(simple_a1b), np.array(simple_a1l), self.cof, 0.3
        )
        return simprize_prob

    def extension(self, simple, X) -> np.ndarray:
        X_0 = np.full((self.n_all), 0.0, dtype=float)
        sa1h, sa1b, sa1l = simple.plains()
        for ih in range(self.n_h):
            sh = lnearst_idx(sa1h, self.a2h[ih])
            for jb in range(self.n_b):
                sb = lnearst_idx(sa1b, self.a2b[jb])
                for kl in range(self.n_l):
                    sl = lnearst_idx(sa1l, self.a2l[kl])
                    idx = kl + jb * self.n_l + ih * self.n_b * self.n_l
                    sidx = sl + sb * simple.n_l + sh * simple.n_b * simple.n_l
                    X_0[idx] = X[sidx]

        return X_0


class INPUT:
    tec: np.ndarray = np.array([])
    sat: list[XYZ] = []
    rec: list[XYZ] = []

    n_obs: int = 0

    def __init__(self, tec, sat, rec):
        self.tec = tec
        self.sat = sat
        self.rec = rec
        self.n_obs = self.tec.shape[0]


# # # # # # # # # # # # # # # # # # # #

# # Import observation File # # # # # #


def ImportDataFromTomoI(exp: EXPERIMENT) -> INPUT:
    country = exp.c
    year4 = exp.y
    day = exp.d
    ep = exp.ep

    Tomography_input = "{dr}/tomoi/{c}/{y:04d}/{d:03d}/{code}/{ep:04d}.tomoi".format(
        dr=drive, c=country, y=year4, d=day, code="_______________", ep=ep
    )

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
                sat = XYZ(
                    float(line.split()[1]),
                    float(line.split()[2]),
                    float(line.split()[3]),
                )
                sats.append(sat)
                rec = XYZ(
                    float(line.split()[4]),
                    float(line.split()[5]),
                    float(line.split()[6]),
                )
                recs.append(rec)

    return INPUT(np.array(data, dtype=float), sats, recs)


# # # # # # # # # # # # # # # # # # # #

# # Calculation C # # # # # # # # # # #


def Generate_C(
    exp: EXPERIMENT, Input: INPUT, setting: SETTING, write=True
) -> csr_matrix:
    country = exp.c
    year4 = exp.y
    day = exp.d
    code = exp.code
    ep = exp.ep

    n_obs = Input.n_obs
    n_all = setting.n_all

    lsat = Input.sat
    lrec = Input.rec

    n_all, n_h, n_b, n_l = setting.nbrock()

    os.makedirs(
        "{dr}/tomoc/{c}/{y:04d}/{d:03d}/{code}".format(
            dr=drive, c=country, y=year4, d=day, code=code
        ),
        exist_ok=True,
    )
    Tomography_C = "{dr}/tomoc/{c}/{y:04d}/{d:03d}/{code}/{epc:04d}.tomoc".format(
        dr=drive, c=country, y=year4, d=day, code=code, epc=ep
    )

    A = lil_matrix((n_obs, n_all))
    if write:
        with open(Tomography_C, "w") as f:
            print(n_obs, n_all, file=f)
            for iobs in tqdm(range(n_obs)):
                A_idx, A_data = Generate_C_row(lsat[iobs], lrec[iobs], setting)
                for jidx in range(len(A_idx)):
                    A[iobs, A_idx[jidx]] = A_data[jidx]
                    print(iobs, A_idx[jidx], A_data[jidx], file=f)
    else:
        for iobs in tqdm(range(n_obs)):
            A_idx, A_data = Generate_C_row(lsat[iobs], lrec[iobs], setting)
            for jidx in range(len(A_idx)):
                A[iobs, A_idx[jidx]] = A_data[jidx]

    return csr_matrix(A)


# return IRI value at hgt[km] -> TECU/km


def inner(x1: XYZ, x2: XYZ) -> float:
    ans = 0.0
    ans += x1.x * x2.x
    ans += x1.y * x2.y
    ans += x1.z * x2.z
    return ans


# return zenith theta[rad]


def zenith(rec: XYZ, sat: XYZ, H: float):
    t, ipp = spH(rec, sat, H)
    cosZ = inner(ipp - rec, ipp) / math.sqrt(
        inner(ipp, ipp) * inner(ipp - rec, ipp - rec)
    )
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
            ans[ib, jl] = (
                iri2016.IRI(day, [H, H, 1], bs[ib, jl], ls[ib, jl]).ne[0] * 1.0e3
            )
    return ans


def _interpolate(ans, m_B, M_B, m_L, M_L, nb, nl, lat, lon):
    db = (M_B - m_B) / (nb - 1)
    dl = (M_L - m_L) / (nl - 1)
    i_b = math.floor((lat - m_B) / db)
    i_l = math.floor((lon - m_L) / dl)
    bef_b = m_B + db * i_b
    aft_b = m_B + db * (i_b + 1)
    bef_l = m_L + dl * i_l
    aft_l = m_L + dl * (i_l + 1)
    Ne = 0.0
    Ne += ((lat - bef_b) / db) * ((lon - bef_l) / dl) * ans[i_b, i_l]
    Ne += ((lat - bef_b) / db) * ((aft_l - lon) / dl) * ans[i_b, i_l + 1]
    Ne += ((aft_b - lat) / db) * ((lon - bef_l) / dl) * ans[i_b + 1, i_l]
    Ne += ((aft_b - lat) / db) * ((aft_l - lon) / dl) * ans[i_b + 1, i_l + 1]
    return Ne


def Del_Plasmaphic(exp: EXPERIMENT, Input: INPUT, setting: SETTING):
    year4 = exp.y
    doy = exp.d
    epoch = exp.ep

    UT = epoch / 120.0
    tmp = datetime.datetime(year=year4, month=1, day=1) + datetime.timedelta(
        days=doy - 1, hours=UT
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

    m_H, M_H, m_B, M_B, m_L, M_L = setting.bounds()
    tecs = Input.tec
    recs = Input.rec
    sats = Input.sat

    n_obs = Input.n_obs

    nb = 7
    nl = 7
    _dIRI = Import_iri2016(day, M_H, m_B, M_B, m_L, M_L, nb, nl)

    tecs_copy = tecs.copy()
    scale = 1.0e3
    for iobs in range(n_obs):
        theta = zenith(recs[iobs], sats[iobs], M_H)
        t, IPP = spH(recs[iobs], sats[iobs], M_H)
        dtec = (
            _interpolate(
                _dIRI, m_B, M_B, m_L, M_L, nb, nl, IPP.to_BLH().b, IPP.to_BLH().l
            )
            * scale
            / 1.0e16
            * abs(math.cos(theta))
        )
        tecs_copy[iobs] -= dtec
    return tecs_copy


# -1 / 0 / 1 / 2 / 3 / 4 / 5 / 6
#    / 0 / 1 / 3 / 5 / 7 / 11
# -3       2                   14
#


def Generate_C_row(sat: XYZ, rec: XYZ, setting: SETTING):
    n_all, n_h, n_b, n_l = setting.nbrock()
    a1h, a1b, a1l = setting.plains()

    M_H = setting.M_H

    sat_blh = sat.to_BLH()
    rec_blh = rec.to_BLH()
    
    s_H = sat_blh.h
    s_B = sat_blh.b
    r_H = rec_blh.h
    r_B = rec_blh.b
    
    
    cp = {}
    C_idx = []
    C_data = []

    tsat, tmpsat = spH(rec, sat, M_H, False,t=(M_H-r_H)/(s_H-r_H))
    for ib in range(n_b + 1):
        tipp, ipp = spB(rec, tmpsat, a1b[ib], False,t=(a1b[ib]-r_B)/(s_B-r_B))
        if abs(ipp.to_BLH().b - a1b[ib]) > 0.1:
            pass
        else:
            cp[tipp * tsat] = ipp
    for il in range(n_l + 1):
        tipp, ipp = spL(rec, sat, a1l[il])
        if abs(ipp.to_BLH().l - a1l[il]) > 0.1:
            pass
        else:
            cp[tipp] = ipp
    for ih in range(n_h + 1):
        tipp, ipp = spH(rec, sat, a1h[ih], False,t=(a1h[ih]-r_H)/(s_H-r_H))
        if abs(ipp.to_BLH().h - a1h[ih]) > 1.0:
            pass
        else:
            cp[tipp] = ipp

    cp_sort = sorted(cp.items())
    list_key = []
    for key in cp_sort:
        list_key.append(key[0])
    for i in range(len(list_key) - 1):
        bkey = list_key[i]
        akey = list_key[i + 1]
        if bkey > 1.005 or akey < -0.005:
            pass
        else:
            bpos: XYZ = cp[bkey]  # XYZ
            apos: XYZ = cp[akey]  # XYZ
            mpos = bpos * 0.5 + apos * 0.5
            mpos_blh = mpos.to_BLH()
            bidx = lnearst_idx(a1b, mpos_blh.b)
            lidx = lnearst_idx(a1l, mpos_blh.l)
            hidx = lnearst_idx(a1h, mpos_blh.h)
            if -1 < bidx < n_b and -1 < lidx < n_l and -1 < hidx < n_h:
                idx = lidx + bidx * n_l + hidx * n_b * n_l
                C_idx.append(idx)
                C_data.append((bpos - apos).L2())

    return C_idx, C_data  # , b_ipp, a_ipp


# [km or deg]
# float return is distance parameter, normalize like
#  Receiver                       Satelite
#    0.0     -> -> -> -> -> ->     1.0

# returnのfloatはRECで0,SATで1となるように標準化した距離パラメータ

# at rec ... t = 0.0
# at sat ... t = 1.0


def spH(rec: XYZ, sat: XYZ, H, debug_mode=False, t=1.0) -> tuple[float, XYZ]:
    if debug_mode:
        print(1, rec.to_BLH(), sat.to_BLH())
    eps = 0.01  # [km]
    dt = 0.01
    dt_inv = 100.0
    point = XYZ(
        rec.x * (1 - t) + sat.x * t,
        rec.y * (1 - t) + sat.y * t,
        rec.z * (1 - t) + sat.z * t,
    )
    if abs(point.to_BLH().h - H) < eps:
        if t < 0.0:
            return -1.0, rec
        elif t > 1.0:
            return 2.0, sat
        else:
            return t, point
    else:
        dpoint = XYZ(
            rec.x * (1 - t - dt) + sat.x * (t + dt),
            rec.y * (1 - t - dt) + sat.y * (t + dt),
            rec.z * (1 - t - dt) + sat.z * (t + dt),
        )
        fbar_i = (dpoint.to_BLH().h - point.to_BLH().h) * dt_inv
        tbar = t - (point.to_BLH().h - H) / fbar_i
        return spH(rec, sat, H, debug_mode, tbar)


def spB(rec: XYZ, sat: XYZ, B, debug_mode=False, t=1.0) -> tuple[float, XYZ]:
    if debug_mode:
        print(2, t, B, (rec * (1 - t) + sat * t).to_BLH())
    # 1.0e-5[deg]=1[km]
    eps = 4.0e-5
    dt = 0.00001
    point = XYZ(
        rec.x * (1 - t) + sat.x * t,
        rec.y * (1 - t) + sat.y * t,
        rec.z * (1 - t) + sat.z * t,
    )
    if t < 0.0 or t > 1.0:
        return -1.0, rec
    if abs(point.to_BLH().b - B) < eps:
        if t < 0.0:
            return -1.0, rec
        elif t > 1.0:
            return 2.0, sat
        else:
            return t, point
    else:
        dpoint = XYZ(
            rec.x * (1 - t - dt) + sat.x * (t + dt),
            rec.y * (1 - t - dt) + sat.y * (t + dt),
            rec.z * (1 - t - dt) + sat.z * (t + dt),
        )
        fbar_i = (dpoint.to_BLH().b - point.to_BLH().b) / dt
        tbar = t - (point.to_BLH().b - B) / fbar_i
        return spB(rec, sat, B, debug_mode, tbar)


def spL(rec: XYZ, sat: XYZ, L) -> tuple[float, XYZ]:
    tanL = math.tan(math.radians(L))
    s = (sat.y - sat.x * tanL) / ((rec.x - sat.x) * tanL - rec.y + sat.y)
    ans = XYZ(0.0, 0.0, 0.0)
    ans.x = s * rec.x + (1 - s) * sat.x
    ans.y = s * rec.y + (1 - s) * sat.y
    ans.z = s * rec.z + (1 - s) * sat.z
    return 1.0 - s, ans


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


def Generate_H(setting: SETTING) -> csr_matrix:
    M, n_h, n_b, n_l = setting.nbrock()
    alpha = setting.alpha
    a2h, a2b, a2l = setting.centers()
    H = lil_matrix((M, M))
    for ih in range(n_h):
        for jb in range(n_b):
            for kl in range(n_l):
                idx = kl + jb * n_l + ih * n_l * n_b
                coeff_sum = 0.0
                # west
                if kl != 0:
                    idx1 = kl - 1 + jb * n_l + ih * n_l * n_b
                    H[idx, idx1] = coeff(a2h[ih]) * -1.0 * alpha
                    coeff_sum -= H[idx, idx1]
                # east
                if kl != n_l - 1:
                    idx2 = kl + 1 + jb * n_l + ih * n_l * n_b
                    H[idx, idx2] = coeff(a2h[ih]) * -1.0 * alpha
                    coeff_sum -= H[idx, idx2]
                # south
                if jb != 0:
                    idx3 = kl + (jb - 1) * n_l + ih * n_l * n_b
                    H[idx, idx3] = coeff(a2h[ih]) * -1.0 * alpha
                    coeff_sum -= H[idx, idx3]
                # north
                if jb != n_b - 1:
                    idx4 = kl + (jb + 1) * n_l + ih * n_l * n_b
                    H[idx, idx4] = coeff(a2h[ih]) * -1.0 * alpha
                    coeff_sum -= H[idx, idx4]
                # below
                if ih != 0:
                    idx5 = kl + jb * n_l + (ih - 1) * n_l * n_b
                    H[idx, idx5] = coeff(a2h[ih - 1]) * -1.0
                    coeff_sum -= H[idx, idx5]
                # above
                if ih < n_h - 1:
                    idx6 = kl + jb * n_l + (ih + 1) * n_l * n_b
                    H[idx, idx6] = coeff(a2h[ih + 1]) * -1.0
                    coeff_sum -= H[idx, idx6]
                if ih == 0 or ih == n_h - 1:
                    H[idx, idx] = coeff_sum + coeff(a2h[ih]) * 1.0
                else:
                    H[idx, idx] = coeff_sum
    return H.tocsr()


def Generate_Y(exp: EXPERIMENT, setting: SETTING):
    n_all, n_h, n_b, n_l = setting.nbrock()
    a1h, a1b, a1l = setting.plains()
    a2h, a2b, a2l = setting.centers()
    m_H, M_H, m_B, M_B, m_L, M_L = setting.bounds()

    year4 = exp.y
    doy = exp.d
    epoch = exp.ep

    Y = np.full((n_all, 1), 0.0, dtype=float)
    UT = epoch / 120.0
    tmp = datetime.datetime(year=year4, month=1, day=1) + datetime.timedelta(
        days=doy - 1, hours=UT
    )
    year = tmp.year
    month = tmp.month
    day = tmp.day
    hour = tmp.hour
    minute = tmp.minute
    second = tmp.second
    sday = "{y:04d}-{m:02d}-{d:02d} {h:02d}:{mn:02d}:{s:02d}".format(
        y=year, m=month, d=day, h=hour, mn=minute, s=second
    )
    nb = 7
    nl = 7
    _dIRI = Import_iri2016(sday, M_H, m_B, M_B, m_L, M_L, nb, nl)
    for jb in range(n_b):
        for kl in range(n_l):
            idx1 = kl + jb * n_l + (n_h - 1) * n_l * n_b
            Y[idx1, 0] = (
                _interpolate(_dIRI, m_B, M_B, m_L, M_L, nb, nl, a2b[jb], a2l[kl])
                / 1.0e16
            )
    return Y


# # # # # # # # # # # # # # # # # # # #

# # Tomography # # # # # # # # # # # # #


def Tomography(
    A: csr_matrix,
    B: np.ndarray,
    H: csr_matrix,
    Y: np.ndarray,
    co: float,
    x_0: np.ndarray,
    First: bool,
) -> np.ndarray:
    if First:
        X = spsolve(A.T * A + co * co * H.T * H, A.T * B + co * co * H.T * Y)
    else:
        X, info = bicgstab(
            A.T * A + co * co * H.T * H,
            A.T * B + co * co * H.T * Y,
            x0=x_0,
            tol=1.0e-7,
            maxiter=5000,
        )

    return X


# # # # # # # # # # # # # # # # # # # #

# # OUTPUT # # # # # # # # # # # # # #


def Write_B(B, exp: EXPERIMENT):
    country = exp.c
    year4 = exp.y
    day = exp.d
    ep = exp.ep

    os.makedirs(
        "{dr}/tomob/{c}/{y:04d}/{d:03d}/{code}".format(
            dr=drive, c=country, y=year4, d=day, ep=ep, code="_______________"
        ),
        exist_ok=True,
    )
    tomob = "{dr}/tomob/{c}/{y:04d}/{d:03d}/{code}/{ep:04d}.tomob".format(
        dr=drive, c=country, y=year4, d=day, ep=ep, code="_______________"
    )

    with open(tomob, "w") as bf:
        print(B.shape[0], file=bf)
        for i in range(B.shape[0]):
            print(i, B[i], file=bf)


def OutPut(X, op_file, epoch, setting: SETTING):
    n_all, n_h, n_b, n_l = setting.nbrock()
    a1h, a1b, a1l = setting.plains()

    # print(X.shape)
    ut = epoch / 120.0
    dut = datetime.timedelta(hours=ut)
    utsec = dut.seconds
    uth = utsec // 3600
    utm = (utsec - uth * 3600) // 60
    uts = utsec % 60
    with open(op_file, "w") as f:
        print("# Tomography Result", file=f)
        print("# ", file=f)
        print("# RUN BY {prog}".format(prog="tomography.py"), file=f)
        print("# ", file=f)
        print("# UTC : {dt}".format(dt=datetime.datetime.now()), file=f)
        print("# ", file=f)
        print("# Number of Boxel (All area)", file=f)
        print("# Latitude : {nb:03d}".format(nb=n_b), file=f)
        print("# Longitude : {nl:03d}".format(nl=n_l), file=f)
        print("# Height : {nh:03d}".format(nh=n_h), file=f)
        print("# ", file=f)
        print("# *** Plain List ***", file=f)
        print("# Latitude", file=f)
        for ib in range(n_b + 1):
            print("# {b:+06.2f}".format(b=a1b[ib]), file=f)
        print("# Longitude", file=f)
        for il in range(n_l + 1):
            print("# {l:+06.2f}".format(l=a1l[il]), file=f)
        print("# Height", file=f)
        for ih in range(n_h + 1):
            print("# {h:+07.2f}".format(h=a1h[ih]), file=f)
        print("# ", file=f)
        print("# END OF HEADER", file=f)
        print("", file=f)
        for ih in range(n_h):
            for jb in range(n_b):
                for kl in range(n_l):
                    if True:
                        f.write(
                            "{teq:+15.13f} ".format(
                                teq=X[kl + jb * n_l + ih * n_l * n_b]
                            )
                        )
                    # else:
                    #     f.write(
                    #         "{teq:+15.13f} ".format(teq=np.nan))
                print("", file=f)
            print("", file=f)


# # # # # # # # # # # # # # # # # # # #

# # IMPORT # # # # # # # # # # # # # #


def Read_A(exp: EXPERIMENT) -> csr_matrix:
    country = exp.c
    year4 = exp.y
    day = exp.d
    code = exp.code
    ep = exp.ep
    tomoc = "{dr}/tomoc/{c}/{y:04d}/{d:03d}/{cd}/{ep:04d}.tomoc".format(
        dr=drive, c=country, y=year4, d=day, cd=code, ep=ep
    )
    with open(tomoc, "r") as f:
        line = f.readline()
        L = int(line.split()[0])
        M = int(line.split()[1])
        A = lil_matrix((L, M))
        while True:
            line = f.readline()
            if not line:
                break
            else:
                i = int(line.split()[0])
                j = int(line.split()[1])
                l = float(line.split()[2])
                A[i, j] = l
    return csr_matrix(A)


def Read_B(exp: EXPERIMENT) -> np.ndarray:
    country = exp.c
    year4 = exp.y
    day = exp.d
    code = exp.code
    ep = exp.ep
    tomob = "{dr}/tomob/{c}/{y:04d}/{d:03d}/{cd}/{ep:04d}.tomob".format(
        dr=drive, c=country, y=year4, d=day, cd="_______________", ep=ep
    )

    with open(tomob, "r") as f:
        line = f.readline()
        L = int(line.split()[0])
        B = np.full((L, 1), 0.0, dtype=float)
        while True:
            line = f.readline()
            if not line:
                break
            i = int(line.split()[0])
            b = float(line.split()[1])
            B[i, 0] = b

    return B


def Read_X(exp: EXPERIMENT) -> np.ndarray:
    country = exp.c
    year4 = exp.y
    day = exp.d
    code = exp.code
    ep = exp.ep
    tomo = "{dr}/tomo/{c}/{y:04d}/{d:03d}/{cd}/{ep:04d}.tomo".format(
        dr=drive, c=country, y=year4, d=day, cd=code, ep=ep
    )
    with open(tomo, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if "END OF HEADER" in line:
                break
            if "Number of Boxel (All area)" in line:
                line = f.readline()
                n_b = int(line.split()[3])
                line = f.readline()
                n_l = int(line.split()[3])
                line = f.readline()
                n_h = int(line.split()[3])

        n_all = n_b * n_l * n_h
        datas = np.full((n_all, 1), 0.0, dtype=float)
        for kh in range(n_h):
            line = f.readline()
            for jb in range(n_b):
                line = f.readline()
                for il in range(n_l):
                    datas[kh * n_b * n_l + jb * n_l + il, 0] = float(line.split()[il])
    return datas


# # # # # # # # # # # # # # # # # # # #


def Setting_Process(exp: EXPERIMENT, setting: SETTING):
    Input = ImportDataFromTomoI(exp)

    lil_A = Generate_C(exp, Input, setting)

    B = Del_Plasmaphic(exp, Input, setting)

    Write_B(B, exp)


def Solve_First_Epoch(exp: EXPERIMENT, setting: SETTING):
    country = exp.c
    year4 = exp.y
    day = exp.d
    code = exp.code
    ep = exp.ep

    n_all = setting.n_all
    cof = setting.cof

    if n_all < 30000:
        # Solve Original
        csr_A = Read_A(exp)
        nd_B = Read_B(exp)
        csr_H = setting.H
        nd_Y = Generate_Y(exp, setting)

        co = float(csr_matrix.trace(csr_A.T * csr_A) / setting.trH_2)

        X = Tomography(csr_A, nd_B, csr_H, nd_Y, cof * co, x_0=np.nan, First=True)

        os.makedirs(
            "{dr}/tomo/{c}/{y:04}/{d:03}/{code}".format(
                dr=drive, c=country, y=year4, d=day, code=code
            ),
            exist_ok=True,
        )
        op_file = "{dr}/tomo/{c}/{y:04}/{d:03}/{code}/{epc:04d}.tomo".format(
            dr=drive, c=country, y=year4, d=day, code=code, epc=ep
        )
        OutPut(X, op_file, ep, setting)

    else:
        # Solve Simprize Problem
        N_b = 35
        N_l = 35
        simple_setting = setting.simprize(N_b, N_l)

        Input = ImportDataFromTomoI(exp)
        scsr_A = Generate_C(exp, Input, simple_setting, write=False)
        snd_B = Read_B(exp)
        scsr_H = Generate_H(simple_setting)
        snd_Y = Generate_Y(exp, simple_setting)

        sco = float(
            csr_matrix.trace(scsr_A.T * scsr_A) / csr_matrix.trace(scsr_H.T * scsr_H)
        )

        X = Tomography(scsr_A, snd_B, scsr_H, snd_Y, cof * sco, x_0=np.nan, First=True)

        X_0 = setting.extension(simple_setting, X)

        csr_A = Read_A(exp)
        nd_B = Read_B(exp)
        csr_H = setting.H
        nd_Y = Generate_Y(exp, setting)
        co = float(csr_matrix.trace(csr_A.T * csr_A) / setting.trH_2)

        X = Tomography(csr_A, nd_B, csr_H, nd_Y, cof * co, x_0=X_0, First=False)

        os.makedirs(
            "{dr}/tomo/{c}/{y:04}/{d:03}/{code}".format(
                dr=drive, c=country, y=year4, d=day, code=code
            ),
            exist_ok=True,
        )
        op_file = "{dr}/tomo/{c}/{y:04}/{d:03}/{code}/{epc:04d}.tomo".format(
            dr=drive, c=country, y=year4, d=day, code=code, epc=ep
        )

        OutPut(X, op_file, ep, setting)


def Solving_Process(exp: EXPERIMENT, setting: SETTING, eps, idx):
    country = exp.c
    year4 = exp.y
    day = exp.d
    code = exp.code
    ep = exp.ep

    cof = setting.cof

    csr_A = Read_A(exp)
    nd_B = Read_B(exp)
    csr_H = setting.H
    nd_Y = Generate_Y(exp, setting)
    co = float(csr_matrix.trace(csr_A.T * csr_A) / setting.trH_2)

    latest = max(0, idx - 1)
    exp_latest = EXPERIMENT(country, year4, day, code, eps[latest])

    X_0 = Read_X(exp_latest)

    X = Tomography(csr_A, nd_B, csr_H, nd_Y, cof * co, X_0, First=False)

    os.makedirs(
        "{dr}/tomo/{c}/{y:04}/{d:03}/{code}".format(
            dr=drive, c=country, y=year4, d=day, code=code
        ),
        exist_ok=True,
    )
    op_file = "{dr}/tomo/{c}/{y:04}/{d:03}/{code}/{epc:04d}.tomo".format(
        dr=drive, c=country, y=year4, d=day, code=code, epc=ep
    )
    OutPut(X, op_file, ep, setting)

    print(exp, "end")


# Main
# Ax -> B
# Hx -> Y


def __Tomography__(
    country: str,
    year4: int,
    day: int,
    CODE: str,
    eps: np.ndarray,
    setting: SETTING,
    M_w: int,
):
    print("** Start Tomography **")
    start = time.time()

    n_t = eps.shape[0]
    n_all, n_h, n_b, n_l = setting.nbrock()

    exps = []
    for i in range(n_t):
        exps.append(EXPERIMENT(country, year4, day, CODE, eps[i]))

    csr_H = Generate_H(setting)
    setting.__initH__(csr_H)

    # 131 x 122 x 26 -> 13 sec
    print("** Common Setting Complete **", time.time() - start)

    for exp in exps:
        Setting_Process(exp, setting)
        discord = Discord(
            url="https://discord.com/api/webhooks/1126067270238605352/By5LHKmX15j-A0tpiDHUsphD3HPWvgZy6l9O2Md7DJXLin9QWLXzuBusswRcN7CTeNHD")
        discord.post(content="{y:04d} {d:03d} {ep:04d} setting end : {dt}".format(
            y=exp.y, d=exp.d, ep=exp.ep, dt=time.time()-start))

    print("** Formula Setting Complete **", time.time() - start)

    Solve_First_Epoch(exps[0], setting)
    discord = Discord(
            url="https://discord.com/api/webhooks/1126067270238605352/By5LHKmX15j-A0tpiDHUsphD3HPWvgZy6l9O2Md7DJXLin9QWLXzuBusswRcN7CTeNHD")
    discord.post(content="{y:04d} {d:03d} {ep:04d} solved : {dt}".format(
            y=exps[0].y, d=exps[0].d, ep=exps[0].ep, dt=time.time()-start))

    print("** Solve First Problem **", time.time() - start)

    for idx, exp in enumerate(exps[1:]):
        Solving_Process(exp, setting, exps, idx)
        discord = Discord(
            url="https://discord.com/api/webhooks/1126067270238605352/By5LHKmX15j-A0tpiDHUsphD3HPWvgZy6l9O2Md7DJXLin9QWLXzuBusswRcN7CTeNHD")
        discord.post(content="{y:04d} {d:03d} {ep:04d} solved : {dt}".format(
            y=exp.y, d=exp.d, ep=exp.ep, dt=time.time()-start))

    print("** All Problem Completed**", time.time() - start)


if __name__ == "__main__":
    country = "jp"
    year4 = 2016
    day = 193
    CODE = "MSTID_1+2_025d_3_2_0_8-1"
    eps = np.arange(1600, 2200, 1)
    # n_h = 26
    a1h = np.array(
        [
            070.0, 100.0, 130.0, 160.0, 190.0, 220.0, 250.0, 280.0, 310.0, 340.0,
            370.0, 400.0, 430.0, 460.0, 490.0, 520.0, 550.0, 580.0, 610.0, 660.0,
            710.0, 760.0, 810.0, 860.0, 910.0, 960.0, 1010.0
        ]
    )
    # n_l = 131
    a1l = np.array([095.00, 097.00, 099.00, 101.00, 103.00, 105.00, 106.00, 107.00, 108.00, 109.00,
                    110.00, 111.00, 112.00, 113.00, 114.00, 115.00, 116.00, 117.00, 118.00, 119.00,
                    120.00, 121.00, 122.00, 123.00, 124.00, 125.00, 125.25, 125.50, 125.75, 126.00,
                    126.25, 126.50, 126.75, 127.00, 127.25, 127.50, 127.75, 128.00, 128.25, 128.50,
                    128.75, 129.00, 129.25, 129.50, 129.75, 130.00, 130.25, 130.50, 130.75, 131.00,
                    131.25, 131.50, 131.75, 132.00, 132.25, 132.50, 132.75, 133.00, 133.25, 133.50,
                    133.75, 134.00, 134.25, 134.50, 134.75, 135.00, 135.25, 135.50, 135.75, 136.00,
                    136.25, 136.50, 136.75, 137.00, 137.25, 137.50, 137.75, 138.00, 138.25, 138.50,
                    138.75, 139.00, 139.25, 139.50, 139.75, 140.00, 140.25, 140.50, 140.75, 141.00,
                    141.25, 141.50, 141.75, 142.00, 142.25, 142.50, 142.75, 143.00, 143.25, 143.50,
                    143.75, 144.00, 144.25, 144.50, 144.75, 145.00, 146.00, 147.00, 148.00, 149.00,
                    150.00, 151.00, 152.00, 153.00, 154.00, 155.00, 156.00, 157.00, 158.00, 159.00,
                    160.00, 161.00, 162.00, 163.00, 164.00, 165.00, 167.00, 169.00, 171.00, 173.00,
                    175.00, 177.00])
    # n_b = 122
    a1b = np.array([-5.00, -3.00, -1.00, 01.00, 03.00, 05.00, 07.00, 09.00, 11.00, 13.00,
                    15.00, 16.00, 17.00, 18.00, 19.00, 20.00, 21.00, 22.00, 23.00, 24.00,
                    25.00, 25.25, 25.50, 25.75, 26.00, 26.25, 26.50, 26.75, 27.00, 27.25,
                    27.50, 27.75, 28.00, 28.25, 28.50, 28.75, 29.00, 29.25, 29.50, 29.75,
                    30.00, 30.25, 30.50, 30.75, 31.00, 31.25, 31.50, 31.75, 32.00, 32.25,
                    32.50, 32.75, 33.00, 33.25, 33.50, 33.75, 34.00, 34.25, 34.50, 34.75,
                    35.00, 35.25, 35.50, 35.75, 36.00, 36.25, 36.50, 36.75, 37.00, 37.25,
                    37.50, 37.75, 38.00, 38.25, 38.50, 38.75, 39.00, 39.25, 39.50, 39.75,
                    40.00, 40.25, 40.50, 40.75, 41.00, 41.25, 41.50, 41.75, 42.00, 42.25,
                    42.50, 42.75, 43.00, 43.25, 43.50, 43.75, 44.00, 44.25, 44.50, 44.75,
                    45.00, 46.00, 47.00, 48.00, 49.00, 50.00, 51.00, 52.00, 53.00, 54.00,
                    55.00, 57.00, 59.00, 61.00, 63.00, 65.00, 67.00, 69.00, 71.00, 73.00,
                    75.00, 77.00, 79.00])
    
    setting = SETTING(a1h=a1h, a1b=a1b, a1l=a1l, cof=1.0e2, alpha=0.8)

    __Tomography__(country, year4, day, CODE, eps, setting, 2)
