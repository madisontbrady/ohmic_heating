import numpy as np
import pandas as pd
from scipy.integrate import solve_bvp, odeint
import scipy
import matplotlib.pyplot as plt
import gastli.Coupling as cpl
import gastli.constants as cte
from scipy.interpolate import CubicSpline
from scipy.interpolate import LinearNDInterpolator
import radvel
import sympy


# Constants, in mks.
AU=1.496e11
RSOLAR = 6.955e8
MU0=1.256637e-6
SB=5.67037e-8
RJUP=7.149e7
MJUP=1.899e27
G=6.67e-11
MEARTH=5.972e24
AMU=1.6605402e-27
MH = 1.008*AMU
MHE = 4.0026*AMU
H=6.626e-34
ME=9.10938e-31
KB=1.380649e-23



# Read in EOS tables from Saumon+1995 (included in the MESA files)

def read_scvh_file(fname):
    f = open(fname)
    lines = f.readlines()
    f.close()
    
    n_values = np.array([len(i.split()) for i in lines])
    n_temps = len(n_values[n_values==2])
    n_models = len(n_values[n_values==n_values[1]])

    data_arr = np.zeros((len(n_values[n_values==n_values[1]]), n_values[1]+1))
    current_temp=np.nan
    j=0
    for i in range(len(lines)):
        values = lines[i].split()
        if len(values)==2:
            current_temp=float(values[0])
        else:
            data_arr[j] = np.concatenate(([current_temp], np.array(values).astype(float)))
            j+=1
    return data_arr
        
fname="h_tab_S95.dat"
h_data = read_scvh_file(fname)

h_logt = h_data[:,0]
h_logp = h_data[:,1]
h_xh2 = h_data[:, 2]
h_xh = h_data[:, 3]
h_logrho = h_data[:, 4]

calcxh2 = LinearNDInterpolator(np.array([h_logt, h_logp]).T, np.array(h_xh2))
calcxh = LinearNDInterpolator(np.array([h_logt, h_logp]).T, np.array(h_xh))
calclogrhoh = LinearNDInterpolator(np.array([h_logt, h_logp]).T, np.array(h_logrho))


fname="he_tab_S95.dat"
he_data = read_scvh_file(fname)

he_logt = he_data[:,0]
he_logp = he_data[:,1]
he_xhe = he_data[:, 2]
he_xhep = he_data[:, 3]
he_logrho = he_data[:, 4]

calcxhe = LinearNDInterpolator(np.array([he_logt, he_logp]).T, np.array(he_xhe))
calcxhep = LinearNDInterpolator(np.array([he_logt, he_logp]).T, np.array(he_xhep))
calclogrhohe = LinearNDInterpolator(np.array([he_logt, he_logp]).T, np.array(he_logrho))




# from https://arxiv.org/pdf/1208.0933

def nn_ne(logt, logp, y=0.283):
    """
    Use the prescription from Chang+2012 to estimate nn/ne, the fraction
    of neutral atoms to electrons.
    
    logt = log temperature (K)
    logp = log pressure (bayres) 
    
    """
    xh2 = calcxh2(logt, logp)
    xh =  calcxh(logt, logp)
    rhoh_theoretical =  10**calclogrhoh(logt, logp) * 1000 # convert to kg/m^3
    xhe = calcxhe(logt, logp)
    xhep = calcxhep(logt, logp)
    rhohe_theoretical=10**calclogrhohe(logt, logp)* 1000 # convert to kg/m^3

    try:
        xh_elec = np.nanmax(np.array([np.zeros(len(xh2)), 1/2 * (1- xh2 - xh)]), axis=0)
        xhe_elec = np.nanmax(np.array([np.zeros(len(xh2)), 1/3 * (2 - 2*xhe - xhep)]),axis=0)
    except:
        xh_elec = np.nanmax(np.array([0, 1/2 * (1- xh2 - xh)]))
        xhe_elec = np.nanmax(np.array([0, 1/3 * (2 - 2*xhe - xhep)]))     
    xhp=xh_elec

    xhe2p = (xhe_elec - xhep)/2

    try:
        xh_elec[xh_elec==0]=np.nan
        xhe_elec[xhe_elec==0]=np.nan
    except:
        if xh_elec==0: xh_elec=np.nan
        if xhe_elec==0: xhe_elec=np.nan

    rho = (((1-y)/rhoh_theoretical) + (y/rhohe_theoretical))**(-1)

    rhoh= (1-y)*rho
    rhohe=y*rho
    

    npr = 3*rhohe/MHE / (1 + 2*xhe + xhep) * (xhe2p +xhep) +  2*rhoh/MH / (1 + 3*xh2 + xh)*xhp
    nn = 3*rhohe/MHE / (1 + 2*xhe + xhep) * (xhe) + 2*rhoh/MH / (1 + 3*xh2 + xh)*xh2 +  2*rhoh/MH / (1 + 3*xh2 + xh)*xh
    ne = 3*rhohe/MHE / (1 + 2*xhe + xhep) * (xhe_elec) + 2*rhoh/MH / (1 + 3*xh2 + xh)*xh_elec
    n = 3*rhohe/MHE / (1 + 2*xhe + xhep) + 2*rhoh/MH / (1 + 3*xh2 + xh)
    return(nn/ne)



def diffusivity_hhe_si(t,p,y=0.283):
    """
    Calculate the diffusivity (m^2/s) using the equation from Blaes & Balbus 1994.
    t = temperature (K)
    p = Pressure (pa) 
    """
    # we multiply pressure by 10 to convert it to bayres for the nn_ne calculation (the tables from Saumon are in cgs)
    eta = 0.023 * nn_ne(np.log10(t), np.log10(p*10), y=y) * t**(1/2) # s^-1

    return eta

def diffusivity_L08(t,p):
    """
    Calculate the diffusivity (m^2/s) using the equation from Laine+2008.
    t = temperature (K)
    p = Pressure (pa) 
    """
    return 1.28e-2 * np.sqrt(p)/t**(3/4) * np.exp(78909/t)
    





def arctan2(y, x):
    # 4-quadrant arctangent
    try:
        val = np.arctan(y / x)
        val[((x < 0) & (y >= 0))] += np.pi
        val[((x < 0) & (y < 0))] -= np.pi
        val[((x == 0) & (y > 0))] = np.pi / 2
        val[((x == 0) & (y < 0))] = -np.pi / 2
        val[((x == 0) & (y == 0))] = np.nan
        return val

    except:
        if x > 0:
            return np.arctan(y / x)
        if x < 0 and y >= 0:
            return np.arctan(y / x) + np.pi
        if x < 0 and y < 0:
            return np.arctan(y / x) - np.pi
        if x == 0 and y > 0:
            return np.pi / 2
        if x == 0 and y < 0:
            return -np.pi / 2




# source - https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf


class Planet:
    """Used to estimate a planet's location in its orbit as a function of time."""
    def __init__(self, t0=0, m0=0, ecc=0, a=1, omega=0, long=0, inc=0, ms=1):
        self.t0 = t0
        self.m0 = m0
        self.ecc = ecc
        self.a = a  # AU
        self.omega = omega  # radians
        self.long = long  # radians
        self.inc = inc  # radians
        self.ms = ms  # msolar
        self.p = np.sqrt(self.a**3 / self.ms) * 365

    def calc_pos(self, t):
        """
        calculate position and velocity of planet at given time t

        """
        mu = G * self.ms * 1.989e33

        dt = 86400 * (t - self.t0)
        m = (self.m0 + dt * np.sqrt(mu / (self.a * 1.496e13) ** 3)) % (2 * np.pi)
        if type(m) != type(np.array([])):
            m = np.array([m])

        eanom = radvel.kepler.kepler(m, np.ones(len(m)) * self.ecc)
        if type(eanom) != type(np.array([])):
            eanom = np.array([eanom])
        vanom = 2 * arctan2(
            np.sqrt(1 + self.ecc) * np.sin(eanom / 2),
            np.sqrt(1 - self.ecc) * np.cos(eanom / 2),
        )
        rc = self.a * (1 - self.ecc * np.cos(eanom))
        pos = (
            rc
            * 1.496e13
            * np.array([np.cos(vanom), np.sin(vanom), np.zeros(len(vanom))])
        )
        vel = (
            np.sqrt(mu * self.a * 1.496e13)
            / (rc * 1.496e13)
            * np.array(
                [
                    -np.sin(eanom),
                    np.sqrt(1 - self.ecc**2) * np.cos(eanom),
                    np.zeros(len(vanom)),
                ]
            )
        )
        pos_inertial = np.array(
            [
                pos[0]
                * (
                    np.cos(self.omega) * np.cos(self.long)
                    - np.sin(self.omega) * np.cos(self.inc) * np.sin(self.long)
                )
                - pos[1]
                * (
                    np.sin(self.omega) * np.cos(self.long)
                    + np.cos(self.omega) * np.cos(self.inc) * np.sin(self.long)
                ),
                pos[0]
                * (
                    np.cos(self.omega) * np.sin(self.long)
                    + np.sin(self.omega) * np.cos(self.inc) * np.cos(self.long)
                )
                + pos[1]
                * (
                    np.cos(self.omega) * np.cos(self.inc) * np.cos(self.long)
                    - np.sin(self.omega) * np.sin(self.long)
                ),
                pos[0] * (np.sin(self.omega) * np.sin(self.inc))
                + pos[1] * (np.cos(self.omega) * np.sin(self.inc)),
            ]
        )
        vel_inertial = np.array(
            [
                vel[0]
                * (
                    np.cos(self.omega) * np.cos(self.long)
                    - np.sin(self.omega) * np.cos(self.inc) * np.sin(self.long)
                )
                - vel[1]
                * (
                    np.sin(self.omega) * np.cos(self.long)
                    + np.cos(self.omega) * np.cos(self.inc) * np.sin(self.long)
                ),
                vel[0]
                * (
                    np.cos(self.omega) * np.sin(self.long)
                    + np.sin(self.omega) * np.cos(self.inc) * np.cos(self.long)
                )
                + vel[1]
                * (
                    np.cos(self.omega) * np.cos(self.inc) * np.cos(self.long)
                    - np.sin(self.omega) * np.sin(self.long)
                ),
                vel[0] * (np.sin(self.omega) * np.sin(self.inc))
                + vel[1] * (np.cos(self.omega) * np.sin(self.inc)),
            ]
        )
        return pos_inertial, vel_inertial
        
        
        

def system(r, y, w, l, rp, eta_func, eta_mult=1, temp_mult=1):
    """helper function for solving for the Yl functions numerically.
    inputs:
    r = Radius to evaluate function at (in meters)
    y = Array of y functions.
    w = frequency (in s^-1)
    l = degree of function (integer)
    rp = Total radius of planet (in meters)
    """
    y1, y2, y3, y4 = y
    dydt = [
        y3,
        y4,
        -w / eta_func(r, rp, eta_mult=eta_mult, temp_mult=temp_mult) * y2
        + l * (l + 1) / r**2 * y1,
        w / eta_func(r, rp, eta_mult=eta_mult, temp_mult=temp_mult) * y1
        + l * (l + 1) / r**2 * y2,
    ]

    return dydt


def bc(y0, yrp, w, l, rp, r0):
    """boundary conditions for the Yl problem.
    inputs:
    y0 = y function values at r = r0
    yrp = y function at r = rp
    w = frequency (in s^-1)
    l = degree of function (integer)
    rp = Total radius of planet (in meters)
    r0 = small-but-nonzero value describing radius of y0 value (in meters)
    """

    # boundary at r = r0, where r0 is small but nonzero
    y10, y20, y30, y40 = y0

    # boundary at r = rp, radius of planet
    y1rp, y2rp, y3rp, y4rp = yrp

    zeroes = [
        y30 - (l + 1) / r0 * y10,
        y40 - (l + 1) / r0 * y20,
        y3rp + l / rp * y1rp - (2 * l + 1) * rp**l,
        y4rp + l / rp * y2rp,
    ]

    return zeroes


def find_ys(w, rp, eta_func, eta_mult=1, temp_mult=1, r0=0.0001, verbose=False):
    """
    Finds the Y1, Y2, Y3, and Y4 as defined in Laine 2008.
    w = frequency (in rad/s), usually the difference between stellar
    rotation frequency and the orbital frequency of the planet.
    rp = radius of planet, in meters.
    r0 = minimum radius considered, in meters.  Usually can just be
    kept at default.
    eta_func = a function of the form eta_func(r, rp) that gives the
    magnetic diffusivity of the planet (in MKS units) as a function of
    radius r (in meters).
    """

    r = np.linspace(r0, rp, 101)


    ystart1 = np.array(
        [
            np.linspace(0, 1.0 * 10**15, len(r)),
            np.linspace(0, 1.0 * -(10**20), len(r)),
            np.linspace(0, 3 * rp - 1 / rp * 1.0 * 10**15, len(r)),
            np.linspace(0, 3 * rp - 1 / rp * 1.0 * -(10**-20), len(r)),
        ]
    )
    ystart2 = np.array(
        [
            np.linspace(0, 1.0 * 10**15, len(r)),
            np.linspace(0, 1.0 * -(10**20), len(r)),
            np.linspace(0, 5 * rp**2 - 2 / rp * 1.0 * 10**15, len(r)),
            np.linspace(0, 5 * rp**2 - 2 / rp * 1.0 * -(10**-20), len(r)),
        ]
    )

    result1 = solve_bvp(
        lambda r, y: system(r, y, w, 1, rp, eta_func, eta_mult, temp_mult),
        lambda y0, yrp: bc(y0, yrp, w, 1, rp, r0),
        r,
        ystart1,
        max_nodes=10000,
    )
    result2 = solve_bvp(
        lambda r, y: system(r, y, w, 2, rp, eta_func, eta_mult, temp_mult),
        lambda y0, yrp: bc(y0, yrp, w, 2, rp, r0),
        r,
        ystart2,
        max_nodes=10000,
    )
    if result1.success != True or result2.success != True and verbose == True:
        print(result1, result2)
        pass

    if True:
        y1x = result1.sol
        y2x = result2.sol


        def y11(r):
            return y1x(r)[0]

        def y12(r):
            return y1x(r)[1]

        def y13(r):
            return y1x(r)[2]

        def y14(r):
            return y1x(r)[3]

        def y21(r):
            return y2x(r)[0]

        def y22(r):
            return y2x(r)[1]

        def y23(r):
            return y2x(r)[2]

        def y24(r):
            return y2x(r)[3]

        ys = {}
        ys["y11"] = y11
        ys["y12"] = y12
        ys["y13"] = y13
        ys["y14"] = y14
        ys["y21"] = y21
        ys["y22"] = y22
        ys["y23"] = y23
        ys["y24"] = y24
        return ys



def eta_simple(r, rp, eta_mult=1, temp_mult=1):
    """
    A simple, nonphysical function from Laine+ 2008 that describes eta as a function
    of radius.  Primarily used for testing.
    r = radius at evaluation (m)
    rp = radius of planet (m)
    eta_mult = multiplicative factor on eta, used to explore how changes in eta influence 
    results.
    temp_mult = multiplicative factor on temperature, does nothing here as temperature
    is not an input into this function.
    """
    return 10**3 * np.e ** (25 * (r / rp) ** 2) * eta_mult


def find_coeffs_zeroecc(rp, y1s, alpha, m, a):

    data = {}

    # l=1, m=1 case.  Standard settings.

    reg11 = y1s["y11"](rp)
    img11 = y1s["y12"](rp)
    reg11dot = y1s["y13"](rp)
    img11dot = y1s["y14"](rp)

    # mu11, mu1m1, nu11, nu1m1, a1, a2, a3, a4
    eqns = np.array(
        [
            [ -img11, img11, -reg11, reg11, 1 / rp * np.sqrt(8 * np.pi / 3) * 1, 0, 0, 0,],
            [ reg11, -reg11, -img11, img11, 0, 1 / rp * np.sqrt(8 * np.pi / 3) * 1, 0, 0,],
            [ -reg11, -reg11, img11, img11, 0, 0, 1 / rp * np.sqrt(8 * np.pi / 3) * 1, 0,],
            [ -img11, -img11,-reg11,-reg11, 0, 0, 0, 1 / rp * np.sqrt(8 * np.pi / 3) * 1,],
            [ -img11dot, img11dot, -reg11dot, reg11dot, -1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1, 0, 0, 0,],
            [ reg11dot, -reg11dot, -img11dot, img11dot, 0, -1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1, 0, 0,],
            [ -reg11dot, -reg11dot, img11dot, img11dot, 0, 0, -1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1, 0,],
            [ -img11dot, -img11dot,-reg11dot,-reg11dot, 0, 0, 0, -1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1,],
        ]
    )

    sols = np.array(
        [
            np.sqrt(8 * np.pi / 3) * 2 * m * np.sin(alpha) * rp**2 / (8 * np.pi * a**3) * 1,
            0,
            0,
            np.sqrt(8 * np.pi / 3) * m * np.sin(alpha) * rp**2 / (8 * np.pi * a**3) * 1,
            np.sqrt(8 * np.pi / 3) * 4 * m * np.sin(alpha) * rp / (8 * np.pi * a**3) * 1,
            0,
            0,
            np.sqrt(8 * np.pi / 3) * 2 * m * np.sin(alpha) * rp / (8 * np.pi * a**3 * 1),
        ]
    )

    x = np.linalg.solve(eqns, sols)

    data["mu111"] = x[0]
    data["mu11m1"] = x[1]
    data["nu111"] = x[2]
    data["nu11m1"] = x[3]

    # l=2, m=0 case.
    reg12 = y1s["y21"](rp)
    img12 = y1s["y22"](rp)
    reg12dot = y1s["y23"](rp)
    img12dot = y1s["y24"](rp)

    # mu120, nu120, beta1, beta2
    eqns = np.array(
        [
            [-img12, -reg12, -np.sqrt(4 * np.pi / 5) * (1 / rp**2), 0],
            [reg12, -img12, 0, -np.sqrt(4 * np.pi / 5) * (1 / rp**2)],
            [-img12dot, -reg12dot, 2 * np.sqrt(4 * np.pi / 5) * (1 / rp**3), 0],
            [reg12dot, -img12dot, 0, 2 * np.sqrt(4 * np.pi / 5) * (1 / rp**3)],
        ]
    )

    sols = np.array(
        [
            np.sqrt(4 * np.pi / 5) * m * np.sin(alpha) * rp**3 / 8 / np.pi / a**4,
            0,
            3 * np.sqrt(4 * np.pi / 5) * m * np.sin(alpha) * rp**2 / 8 / np.pi / a**4,
            0,
        ]
    )

    x = np.linalg.solve(eqns, sols)

    data["mu120"] = x[0]
    data["nu120"] = x[1]

    # mu122, mu12m2, nu122, nu12m2, b1, b2, b3, b4
    eqns = np.array(
        [
            [-img12, -img12, -reg12, -reg12, -4 * np.sqrt(6 * np.pi / 5) * 1 / rp**2, 0, 0, 0,],
            [ reg12,  reg12, -img12, -img12, 0, -4 * np.sqrt(6 * np.pi / 5) * 1 / rp**2, 0, 0,],
            [-reg12,  reg12,  img12, -img12, 0, 0, -4 * np.sqrt(6 * np.pi / 5) * 1 / rp**2, 0,],
            [-img12,  img12, -reg12,  reg12, 0, 0, 0, -4 * np.sqrt(6 * np.pi / 5) * 1 / rp**2,],
            [-img12dot,-img12dot,-reg12dot,-reg12dot, 4 * 2 * np.sqrt(6 * np.pi / 5) * 1 / rp**3, 0, 0, 0,],
            [ reg12dot, reg12dot,-img12dot,-img12dot, 0, 4 * 2 * np.sqrt(6 * np.pi / 5) * 1 / rp**3, 0, 0,],
            [-reg12dot, reg12dot, img12dot,-img12dot, 0, 0, 4 * 2 * np.sqrt(6 * np.pi / 5) * 1 / rp**3, 0,],
            [-img12dot, img12dot,-reg12dot, reg12dot, 0, 0, 0, 4 * 2 ** np.sqrt(6 * np.pi / 5) * 1 / rp**3,],
        ]
    )
    sols = np.array(
        [
            -2 * np.sqrt(6 * np.pi / 5) * m * np.sin(alpha) * rp**3 / 8 / np.pi / a**4,
            0,
            0,
            -4 / 3 * np.sqrt(6 * np.pi / 5) * m * np.sin(alpha) * rp**3 / 8 / np.pi / a**4,
            -3 * 2 * np.sqrt(6 * np.pi / 5) * m * np.sin(alpha) * rp**2 / 8 / np.pi / a**4,
            0,
            0,
            -3 * 4 / 3 * np.sqrt(6 * np.pi / 5) * m * np.sin(alpha) * rp**2 / 8 / np.pi / a**4,
        ]
    )
    x = np.linalg.solve(eqns, sols)

    data["mu122"] = x[0]
    data["mu12m2"] = x[1]
    data["nu122"] = x[2]
    data["nu12m2"] = x[3]

    return data


def find_coeffs(rp, y1s, y2s, y3s, y4s, alpha, m, a, e):
    """find the mu and nu coefficients defined in Laine+ 2008.
    This considers the eccentric case, described in section 3.4 of the paper.
    
    inputs:
    rp = radius of planet (m)
    y1s, y2s, y3s, y4s = outputs of find_ys for the i=1, i=2, i=3, and i=4 cases as defined in 
    section 3.4.2 of the paper.
    alpha = angle of stellar magnetic field (radians)
    a = planet distance from star (m)
    e = planet eccentricity 
    
    """

    data = {}

    # l=1, m=0 case.
    reg21 = y2s["y11"](rp)
    img21 = y2s["y12"](rp)
    reg21dot = y2s["y13"](rp)
    img21dot = y2s["y14"](rp)
    # mu210, nu210, eta1, eta2
    eqns = np.array(
        [
            [reg21, -img21, -1 / rp * np.sqrt(4 * np.pi / 3), 0],
            [-img21, -reg21, 0, -1 / rp * np.sqrt(4 * np.pi / 3)],
            [ reg21dot, -img21dot, 1 / rp**2 * np.sqrt(4 * np.pi / 3), 0],
            [-img21dot, -reg21dot, 0, 1 / rp**2 * np.sqrt(4 * np.pi / 3)],
        ]
    )

    sols = np.array(
        [
            -1 * np.sqrt(4 * np.pi / 3) * m * rp**2 * np.cos(alpha) / 8 / np.pi / a**3 * 3 * e,
            0,
            -2 * np.sqrt(4 * np.pi / 3) * m * rp * np.cos(alpha) / 8 / np.pi / a**3 * 3 * e,
            0,
        ]
    )

    x = np.linalg.solve(eqns, sols)

    data["mu210"] = x[0]
    data["nu210"] = x[1]


    # l=1, m=1 case.  Standard settings.

    reg11 = y1s["y11"](rp)
    img11 = y1s["y12"](rp)
    reg11dot = y1s["y13"](rp)
    img11dot = y1s["y14"](rp)

    # mu11, mu1m1, nu11, nu1m1, a1, a2, a3, a4
    eqns = np.array(
        [
            [ -img11, img11, -reg11, reg11, 1 / rp * np.sqrt(8 * np.pi / 3) * 1, 0, 0, 0,],
            [ reg11, -reg11, -img11, img11, 0, 1 / rp * np.sqrt(8 * np.pi / 3) * 1, 0, 0,],
            [ -reg11, -reg11, img11, img11, 0, 0, 1 / rp * np.sqrt(8 * np.pi / 3) * 1, 0,],
            [ -img11, -img11,-reg11,-reg11, 0, 0, 0, 1 / rp * np.sqrt(8 * np.pi / 3) * 1,],
            [ -img11dot, img11dot, -reg11dot, reg11dot, -1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1, 0, 0, 0,],
            [ reg11dot, -reg11dot, -img11dot, img11dot, 0, -1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1, 0, 0,],
            [ -reg11dot, -reg11dot, img11dot, img11dot, 0, 0, -1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1, 0,],
            [ -img11dot, -img11dot,-reg11dot,-reg11dot, 0, 0, 0, -1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1,],
        ]
    )

    sols = np.array(
        [
            np.sqrt(8 * np.pi / 3) * 2 * m * np.sin(alpha) * rp**2 / (8 * np.pi * a**3) * 1,
            0,
            0,
            np.sqrt(8 * np.pi / 3) * m * np.sin(alpha) * rp**2 / (8 * np.pi * a**3) * 1,
            np.sqrt(8 * np.pi / 3) * 4 * m * np.sin(alpha) * rp / (8 * np.pi * a**3) * 1,
            0,
            0,
            np.sqrt(8 * np.pi / 3) * 2 * m * np.sin(alpha) * rp / (8 * np.pi * a**3 * 1),
        ]
    )

    x = np.linalg.solve(eqns, sols)

    data["mu111"] = x[0]
    data["mu11m1"] = x[1]
    data["nu111"] = x[2]
    data["nu11m1"] = x[3]

    # Now, the cross terms.  Start with i=3

    reg31 = y3s["y11"](rp)
    img31 = y3s["y12"](rp)
    reg31dot = y3s["y13"](rp)
    img31dot = y3s["y14"](rp)

    # mu311, mu31m1, nu311, nu31m1, xi2, xi2, xi6, xi8
    eqns = np.array(
        [
            [-reg31, reg31, img31, -img31, -np.sqrt(8 * np.pi / 3) * 1 / rp, 0, 0, 0],
            [img31, -img31, reg31, -reg31, 0, -np.sqrt(8 * np.pi / 3) * 1 / rp, 0, 0],
            [img31, img31, reg31, reg31, 0, 0, -np.sqrt(8 * np.pi / 3) * 1 / rp, 0],
            [reg31, reg31, -img31, -img31,0, 0, 0, -np.sqrt(8 * np.pi / 3) * 1 / rp],
            [-reg31dot, reg31dot, img31dot, -img31dot, np.sqrt(8 * np.pi / 3) * 1 / rp**2, 0, 0, 0],
            [img31dot, -img31dot, reg31dot, -reg31dot, 0, np.sqrt(8 * np.pi / 3) * 1 / rp**2, 0, 0],
            [img31dot, img31dot, reg31dot, reg31dot, 0, 0, np.sqrt(8 * np.pi / 3) * 1 / rp**2, 0],
            [reg31dot, reg31dot, -img31dot, -img31dot,0, 0, 0, np.sqrt(8 * np.pi / 3) * 1 / rp**2],            
        ]
    )

    sols = np.array(
        [
            0,
            -np.sqrt(8 * np.pi / 3) * 1 * m * rp**2 * np.sin(alpha) / (8 * np.pi * a**3) *  e,
            -np.sqrt(8 * np.pi / 3) * 1 * m * rp**2 * np.sin(alpha) / (8 * np.pi * a**3) * 1/2 * e,
            0,
            0,
            -2 * np.sqrt(8 * np.pi / 3) * 1 * m * rp * np.sin(alpha) / (8 * np.pi * a**3)  * e,
            -2 * np.sqrt(8 * np.pi / 3) * 1 * m * rp * np.sin(alpha) / (8 * np.pi * a**3) * 1/2 * e,
            0,
        ]
    )

    x = np.linalg.solve(eqns, sols)

    data["mu311"] = x[0]
    data["mu31m1"] = x[1]
    data["nu311"] = x[2]
    data["nu31m1"] = x[3]


    # Now, do i=4

    reg41 = y4s["y11"](rp)
    img41 = y4s["y12"](rp)
    reg41dot = y4s["y13"](rp)
    img41dot = y4s["y14"](rp)

    # mu411, mu41m1, nu411, nu41m1, xi1, xi3, xi5, xi7
    eqns = np.array(
        [
            [-reg41, reg41, img41, -img41, -np.sqrt(8 * np.pi / 3) * 1 / rp, 0, 0, 0],
            [img41, -img41, reg41, -reg41, 0, -np.sqrt(8 * np.pi / 3) * 1 / rp, 0, 0],
            [img41, img41, reg41, reg41, 0, 0, -np.sqrt(8 * np.pi / 3) * 1 / rp, 0],
            [reg41, reg41, -img41, -img41,0, 0, 0, -np.sqrt(8 * np.pi / 3) * 1 / rp],
            [-reg41dot, reg41dot, img41dot, -img41dot, np.sqrt(8 * np.pi / 3) * 1 / rp**2, 0, 0, 0],
            [img41dot, -img41dot, reg41dot, -reg41dot, 0, np.sqrt(8 * np.pi / 3) * 1 / rp**2, 0, 0],
            [img41dot, img41dot, reg41dot, reg41dot, 0, 0, np.sqrt(8 * np.pi / 3) * 1 / rp**2, 0],
            [reg41dot, reg41dot, -img41dot, -img41dot,0, 0, 0, np.sqrt(8 * np.pi / 3) * 1 / rp**2],            
        ]
    )

    sols = np.array(
        [
            
            0,
            -np.sqrt(8 * np.pi / 3) * 1 * m * rp**2 * np.sin(alpha) / (8 * np.pi * a**3) *  5*e,
            -np.sqrt(8 * np.pi / 3) * 1 * m * rp**2 * np.sin(alpha) / (8 * np.pi * a**3) * 5/2 * e,
            0,
            0,
            -2 * np.sqrt(8 * np.pi / 3) * 1 * m * rp * np.sin(alpha) / (8 * np.pi * a**3)  * 5*e,
            -2 * np.sqrt(8 * np.pi / 3) * 1 * m * rp * np.sin(alpha) / (8 * np.pi * a**3) * 5/2 * e,
            0,
        ]
    )

    x = np.linalg.solve(eqns, sols)

    data["mu411"] = x[0]
    data["mu41m1"] = x[1]
    data["nu411"] = x[2]
    data["nu41m1"] = x[3]


    return data

def find_coeffs_lam(rp, ys, lam, m, a):
    """find the mu and nu coefficients defined in Laine+ 2008, for the specific case
    where alpha = 0 and eccentricity = 0.
    
    inputs:
    rp = radius of planet (m)
    ys = outputs of find_ys for omega = omega_p
    lam = obliquity angle (radians)
    a = planet distance from star (m)    
    """

    data = {}



    # l=1, m=1 case.  Standard settings.

    reg1 = ys["y11"](rp)
    img1 = ys["y12"](rp)
    reg1dot = ys["y13"](rp)
    img1dot = ys["y14"](rp)

    # mu11, mu1m1, nu11, nu1m1, a1, a2, a3, a4
    eqns = np.array(
        [
            [ -img1, img1, -reg1, reg1, 1 / rp * np.sqrt(8 * np.pi / 3) * 1, 0, 0, 0,],
            [ reg1, -reg1, -img1, img1, 0, 1 / rp * np.sqrt(8 * np.pi / 3) * 1, 0, 0,],
            [ -reg1, -reg1, img1, img1, 0, 0, 1 / rp * np.sqrt(8 * np.pi / 3) * 1, 0,],
            [ -img1, -img1,-reg1,-reg1, 0, 0, 0, 1 / rp * np.sqrt(8 * np.pi / 3) * 1,],
            [ -img1dot, img1dot, -reg1dot, reg1dot, -1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1, 0, 0, 0,],
            [ reg1dot, -reg1dot, -img1dot, img1dot, 0, -1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1, 0, 0,],
            [ -reg1dot, -reg1dot, img1dot, img1dot, 0, 0, -1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1, 0,],
            [ -img1dot, -img1dot,-reg1dot,-reg1dot, 0, 0, 0, -1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1,],
        ]
    )

    sols = np.array(
        [
            0,
            np.sqrt(8 * np.pi / 3) * 2 * m * np.sin(lam) * rp**2 / (8 * np.pi * a**3) * 1,
            np.sqrt(8 * np.pi / 3) * m * np.sin(lam) * rp**2 / (8 * np.pi * a**3) * 1,
            0,
            0,
            np.sqrt(8 * np.pi / 3) * 4 * m * np.sin(lam) * rp / (8 * np.pi * a**3) * 1,
            np.sqrt(8 * np.pi / 3) * 2 * m * np.sin(lam) * rp / (8 * np.pi * a**3 * 1),
            0,
        ]
    )

    x = np.linalg.solve(eqns, sols)

    data["mu111"] = x[0]
    data["mu11m1"] = x[1]
    data["nu111"] = x[2]
    data["nu11m1"] = x[3]

    return data
    
def find_coeffs_lam_alpha(rp, y1s, y2s, y5s, lam, alpha, m, a):
    data = {}
   

    reg1 = y1s["y11"](rp)
    img1 = y1s["y12"](rp)
    reg1dot = y1s["y13"](rp)
    img1dot = y1s["y14"](rp)
    reg2 = y2s["y11"](rp)
    img2 = y2s["y12"](rp)
    reg2dot = y2s["y13"](rp)
    img2dot = y2s["y14"](rp)
    reg5 = y5s["y11"](rp)
    img5 = y5s["y12"](rp)
    reg5dot = y5s["y13"](rp)
    img5dot = y5s["y14"](rp)
    
    # l=1, m=0
    # mu210, nu210, e1, e2
    eqns = np.array(
        [
            [ reg2,    -img2,   -1 / rp * np.sqrt(8 * np.pi / 3) * 1, 0,],
            [ -img2,   -reg2,    0, -1 / rp * np.sqrt(8 * np.pi / 3) * 1,],
            [ reg2dot, -img2dot, 1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1, 0,],
            [ -img2dot,-reg2dot, 0, 1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1,],
        ]
    )

    sols = np.array(
        [
            0,
            np.sqrt(4 * np.pi / 3) * np.sin(alpha) * rp**2 / (8 * np.pi * a**3) * np.sin(lam),
            0,
            2*np.sqrt(4 * np.pi / 3) * np.sin(alpha) * rp / (8 * np.pi * a**3) * np.sin(lam),
        ]
    )

    x = np.linalg.solve(eqns, sols)

    data["mu210"] = x[0]
    data["nu210"] = x[1]
        
    
    
    
    # l=1, m=1
    # mu111, mu11m1, nu111, nu11m1, a1, a2, a3, a4
    eqns = np.array(
        [
            [ img1, -img1, reg1, -reg1, -1 / rp * np.sqrt(8 * np.pi / 3) * 1, 0, 0, 0,],
            [ -reg1, reg1, img1, -img1, 0, -1 / rp * np.sqrt(8 * np.pi / 3) * 1, 0, 0,],
            [ reg1, reg1, -img1, -img1, 0, 0, -1 / rp * np.sqrt(8 * np.pi / 3) * 1, 0,],
            [ img1, img1,reg1,reg1, 0, 0, 0, -1 / rp * np.sqrt(8 * np.pi / 3) * 1,],
            [ img1dot, -img1dot, reg1dot, -reg1dot, 1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1, 0, 0, 0,],
            [ -reg1dot, reg1dot, img1dot,-img1dot, 0, 1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1, 0, 0,],
            [ reg1dot, reg1dot, -img1dot, -img1dot, 0, 0, 1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1, 0,],
            [ img1dot, img1dot,reg1dot,reg1dot, 0, 0, 0, 1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1,],
        ]
    )

    sols = np.array(
        [
            -np.sqrt(8 * np.pi / 3) * np.sin(alpha) * rp**2 / (8 * np.pi * a**3) * (np.cos(lam) + 1),
            0,
            0,
            -np.sqrt(8 * np.pi / 3) * np.sin(alpha) * rp**2 / (8 * np.pi * a**3) * 1/2*(np.cos(lam) + 1),
            -2*np.sqrt(8 * np.pi / 3) * np.sin(alpha) * rp / (8 * np.pi * a**3) * (np.cos(lam) + 1),
            0,
            0,
            -2*np.sqrt(8 * np.pi / 3) * np.sin(alpha) * rp / (8 * np.pi * a**3) * 1/2*(np.cos(lam) + 1),
        ]
    )

    x = np.linalg.solve(eqns, sols)

    data["mu111"] = x[0]
    data["mu11m1"] = x[1]
    data["nu111"] = x[2]
    data["nu11m1"] = x[3]
    
    # mu211, mu21m1, nu211, nu21m1, b1, b2, b3, b4
    eqns = np.array(
        [
            [ img2, -img2, reg2, -reg2, -1 / rp * np.sqrt(8 * np.pi / 3) * 1, 0, 0, 0,],
            [ -reg2, reg2, img2, -img2, 0, -1 / rp * np.sqrt(8 * np.pi / 3) * 1, 0, 0,],
            [ reg2, reg2, -img2, -img2, 0, 0, -1 / rp * np.sqrt(8 * np.pi / 3) * 1, 0,],
            [ img2, img2,reg2,reg2, 0, 0, 0, -1 / rp * np.sqrt(8 * np.pi / 3) * 1,],
            [ img2dot, -img2dot, reg2dot, -reg2dot, 1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1, 0, 0, 0,],
            [ -reg2dot, reg2dot, img2dot,-img2dot, 0, 1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1, 0, 0,],
            [ reg2dot, reg2dot, -img2dot, -img2dot, 0, 0, 1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1, 0,],
            [ img2dot, img2dot,reg2dot,reg2dot, 0, 0, 0, 1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1,],
        ]
    )

    sols = np.array(
        [
            0,
            -np.sqrt(8 * np.pi / 3) * np.cos(alpha) * rp**2 / (8 * np.pi * a**3) * 2*np.sin(lam),
            -np.sqrt(8 * np.pi / 3) * np.cos(alpha) * rp**2 / (8 * np.pi * a**3) * np.sin(lam),
            0,
            0,
            -2*np.sqrt(8 * np.pi / 3) * np.cos(alpha) * rp / (8 * np.pi * a**3) * 2*np.sin(lam),
            -2*np.sqrt(8 * np.pi / 3) * np.cos(alpha) * rp / (8 * np.pi * a**3) * np.sin(lam),
            0,
        ]
    )
    x = np.linalg.solve(eqns, sols)
    data["mu211"] = x[0]
    data["mu21m1"] = x[1]
    data["nu211"] = x[2]
    data["nu21m1"] = x[3]
    
    
    # mu511, mu51m1, nu511, nu51m1, z1, z2, z3, z4
    eqns = np.array(
        [
            [ img5, -img5, reg5, -reg5, -1 / rp * np.sqrt(8 * np.pi / 3) * 1, 0, 0, 0,],
            [ -reg5, reg5, img5, -img5, 0, -1 / rp * np.sqrt(8 * np.pi / 3) * 1, 0, 0,],
            [ reg5, reg5, -img5, -img5, 0, 0, -1 / rp * np.sqrt(8 * np.pi / 3) * 1, 0,],
            [ img5, img5,reg5,reg5, 0, 0, 0, -1 / rp * np.sqrt(8 * np.pi / 3) * 1,],
            [ img5dot, -img5dot, reg5dot, -reg5dot, 1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1, 0, 0, 0,],
            [ -reg5dot, reg5dot, img5dot,-img5dot, 0, 1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1, 0, 0,],
            [ reg5dot, reg5dot, -img5dot, -img5dot, 0, 0, 1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1, 0,],
            [ img5dot, img5dot,reg5dot,reg5dot, 0, 0, 0, 1 / rp**2 * np.sqrt(8 * np.pi / 3) * 1,],
        ]
    )

    sols = np.array(
        [
            0,
            -np.sqrt(8 * np.pi / 3) * np.sin(alpha) * rp**2 / (8 * np.pi * a**3) * (cos(lam)-1),
            -np.sqrt(8 * np.pi / 3) * np.sin(alpha) * rp**2 / (8 * np.pi * a**3) * 1/2*(1-cos(lam)),
            0,
            0,
            -2*np.sqrt(8 * np.pi / 3) * np.sin(alpha) * rp / (8 * np.pi * a**3) * (cos(lam)-1),
            -2*np.sqrt(8 * np.pi / 3) * np.sin(alpha) * rp / (8 * np.pi * a**3) * 1/2*(1-cos(lam)),
            0,
        ]
    )
    x = np.linalg.solve(eqns, sols)
    data["mu511"] = x[0]
    data["mu51m1"] = x[1]
    data["nu511"] = x[2]
    data["nu51m1"] = x[3]

    return data


def find_as(r, y1s, y2s, y3s, y4s, coeffs):
    """
    find the A coefficients as defined in our paper.
    inputs:
    rp = radius of planet (m)
    y1s, y2s, y3s, y4s = outputs of find_ys for the i=1, i=2, i=3, and i=4 cases as defined in 
    section 3.4.2 of the paper.
    coeffs = output of find_coeffs for this model.
    
    """
    reg11 = y1s["y11"](r)
    img11 = y1s["y12"](r)
    reg21 = y2s["y11"](r)
    img21 = y2s["y12"](r)
    reg31 = y3s["y11"](r)
    img31 = y3s["y12"](r)
    reg41 = y4s["y11"](r)
    img41 = y4s["y12"](r)


    a_vals = {}

    a_vals["a11"] = (-coeffs["nu111"] + coeffs["nu11m1"]) * reg11 + (
        -coeffs["mu111"] + coeffs["mu11m1"]
    ) * img11
    a_vals["a12"] = (-coeffs["mu111"] - coeffs["mu11m1"]) * reg11 + (
        coeffs["nu111"] + coeffs["nu11m1"]
    ) * img11
    a_vals["a13"] = (-coeffs["mu111"] + coeffs["mu11m1"]) * reg11 + (
        coeffs["nu111"] - coeffs["nu11m1"]
    ) * img11
    a_vals["a14"] = (coeffs["nu111"] + coeffs["nu11m1"]) * reg11 + (
        coeffs["mu111"] + coeffs["mu11m1"]
    ) * img11

    a_vals["a21"] = coeffs["nu210"] * reg21 + coeffs["mu210"] * img21
    a_vals["a22"] = coeffs["mu210"] * reg21 - coeffs["nu210"] * img21

    a_vals["a31"] = (-coeffs["nu311"] + coeffs["nu31m1"]) * reg11 + (
        -coeffs["mu311"] + coeffs["mu31m1"]
    ) * img11
    a_vals["a32"] = (-coeffs["mu311"] - coeffs["mu31m1"]) * reg11 + (
        coeffs["nu311"] + coeffs["nu31m1"]
    ) * img11
    a_vals["a33"] = (-coeffs["mu311"] + coeffs["mu31m1"]) * reg11 + (
        coeffs["nu311"] - coeffs["nu31m1"]
    ) * img11
    a_vals["a34"] = (coeffs["nu311"] + coeffs["nu31m1"]) * reg11 + (
        coeffs["mu311"] + coeffs["mu31m1"]
    ) * img11

    a_vals["a41"] = (-coeffs["nu411"] + coeffs["nu41m1"]) * reg11 + (
        -coeffs["mu411"] + coeffs["mu41m1"]
    ) * img11
    a_vals["a42"] = (-coeffs["mu411"] - coeffs["mu41m1"]) * reg11 + (
        coeffs["nu411"] + coeffs["nu41m1"]
    ) * img11
    a_vals["a43"] = (-coeffs["mu411"] + coeffs["mu41m1"]) * reg11 + (
        coeffs["nu411"] - coeffs["nu41m1"]
    ) * img11
    a_vals["a44"] = (coeffs["nu411"] + coeffs["nu41m1"]) * reg11 + (
        coeffs["mu411"] + coeffs["mu41m1"]
    ) * img11
    

    return a_vals
    
def find_as_zeroecc(r, ys, coeffs):
    """
    find the A coefficients as defined in our paper, in the specific case that eccentricity = 0.
    inputs:
    rp = radius of planet (m)
    ys = outputs of find_ys
    coeffs = output of find_coeffs for this model.
    
    """
    reg11 = ys["y11"](r)
    img11 = ys["y12"](r)

    reg12 = ys["y21"](r)
    img12 = ys["y22"](r)

    a_vals = {}

    a_vals["a11"] = (-coeffs["nu111"] + coeffs["nu11m1"]) * reg11 + (
        -coeffs["mu111"] + coeffs["mu11m1"]
    ) * img11
    a_vals["a12"] = (-coeffs["mu111"] - coeffs["mu11m1"]) * reg11 + (
        coeffs["nu111"] + coeffs["nu11m1"]
    ) * img11
    a_vals["a13"] = (-coeffs["mu111"] + coeffs["mu11m1"]) * reg11 + (
        coeffs["nu111"] - coeffs["nu11m1"]
    ) * img11
    a_vals["a14"] = (coeffs["nu111"] + coeffs["nu11m1"]) * reg11 + (
        coeffs["mu111"] + coeffs["mu11m1"]
    ) * img11
    try:
        a_vals["a15"] = reg12 * coeffs["nu120"] + img12 * coeffs["mu120"]
        a_vals["a16"] = reg12 * coeffs["mu120"] - img12 * coeffs["nu120"]

        a_vals["a17"] = reg12 * (coeffs["nu122"] + coeffs["nu12m2"]) + img12 * (
            coeffs["mu122"] + coeffs["mu12m2"]
        )
        a_vals["a18"] = reg12 * (coeffs["mu122"] - coeffs["mu12m2"]) + img12 * (
            -coeffs["nu122"] + coeffs["nu12m2"]
        )
        a_vals["a19"] = reg12 * (coeffs["mu122"] + coeffs["mu12m2"]) + img12 * (
            -coeffs["nu122"] - coeffs["nu12m2"]
        )
        a_vals["a110"] = reg12 * (-coeffs["nu122"] + coeffs["nu12m2"]) + img12 * (
            -coeffs["mu122"] + coeffs["mu12m2"]
        )
    except: # an error will occur if only l=1 terms are included
        pass

    return a_vals
    

def pvol_zeroecc(t, w, r, rp, a_vals, eta_func, eta_mult=1, temp_mult=1, l=1):
    """
    Estimates the power-per-volume deposited by magnetic induction, given the A coefficients 
    from find_as_zeroecc().  Only works in the ecc=0 case.
    
    t = times to evaluate model at, in seconds.
    w = frequency difference (2 pi / prot - 2 pi / porb), in s^-1
    r = radius to evaluate model at (m)
    rp = radius of planet (m)
    a_vals = outputs of find_as_zeroecc()
    eta_func = Function that describes eta as a function of r, rp
        form:  eta_func(r, rp, eta_mult, temp_mult)
    eta_mult = multiplicative factor on eta
    temp_mult = multiplicative factor on temperature
    l = spherical harmonic terms to consider,valid values are l=1 or l=2
    """
    
    A11=a_vals["a11"]
    A12=a_vals["a12"]
    A13=a_vals["a13"]
    A14=a_vals["a14"]

    if l==2:
        A15=a_vals["a15"]
        A16=a_vals["a16"]
        A17=a_vals["a17"]
        A18=a_vals["a18"]
        A19=a_vals["a19"]
        A110=a_vals["a110"]
    elif l==1: # A terms from l>1 harmonics are not considered (set to zero)
        A15=0
        A16=0
        A17=0
        A18=0
        A19=0
        A110=0
    else: # l is not a valid value.
        print("Invalid value for max spherical harmonic, l must be 1 or 2")  


    return w**2/eta_func(r, rp, eta_mult=eta_mult, temp_mult=temp_mult) * MU0/r**2 \
    * (np.cos(w*t)**2*(A11**2+A12**2+3*(A17**2+A18**2) + 6*A15**2) 
    +np.sin(w*t)**2*(A13**2+A14**2+3*(A19**2+A110**2) + 6*A16**2) 
    +2*np.cos(w*t)*np.sin(w*t) * (A11*A13+A12*A14+3*(A17*A19+A18*A110)+6*A15*A16)
    )
    
def pvol(t, w, wp, r, rp, a_vals, eta_func, eta_mult=1, temp_mult=1):
    """
    Estimates the power-per-volume deposited by magnetic induction, given the A coefficients 
    from find_as().
    
    t = times to evaluate model at, in seconds.
    w = frequency difference (2 pi / prot - 2 pi / porb), in s^-1
    wp = orbital frequency of planet (2 pi / porb), in s^-1
    r = radius to evaluate model at (m)
    rp = radius of planet (m)
    a_vals = outputs of find_as()
    eta_func = Function that describes eta as a function of r, rp
        form:  eta_func(r, rp, eta_mult, temp_mult)
    eta_mult = multiplicative factor on eta
    temp_mult = multiplicative factor on temperature
    
    """
    A11=a_vals["a11"]
    A12=a_vals["a12"]
    A13=a_vals["a13"]
    A14=a_vals["a14"]
    A21=a_vals["a21"]
    A22=a_vals["a22"]
    A31=a_vals["a31"]
    A32=a_vals["a32"]
    A33=a_vals["a33"]
    A34=a_vals["a34"]
    A41=a_vals["a41"]
    A42=a_vals["a42"]
    A43=a_vals["a43"]
    A44=a_vals["a44"]
 
    return (1/(2*eta_func(r, rp, eta_mult=eta_mult, temp_mult=temp_mult)))/r**2*MU0*(A11**2*w**2
    +A12**2*w**2+A13**2*w**2+A14**2*w**2+A31**2*w**2+A32**2*w**2+A33**2*w**2+A34**2*w**2
    +A41**2*w**2+A42**2*w**2+A43**2*w**2+A44**2*w**2+2*A31**2*w*wp+2*A32**2*w*wp+2*A33**2*w*wp
    +2*A34**2*w*wp-2*A41**2*w*wp-2*A42**2*w*wp-2*A43**2*w*wp-2*A44**2*w*wp+2*A21**2*wp**2
    +2*A22**2*wp**2+A31**2*wp**2+A32**2*wp**2+A33**2*wp**2+A34**2*wp**2+A41**2*wp**2+A42**2*wp**2
    +A43**2*wp**2+A44**2*wp**2+(A11**2*w**2+A12**2*w**2-A13**2*w**2-A14**2*w**2+2*A31*A41*w**2
    +2*A32*A42*w**2-2*A33*A43*w**2-2*A34*A44*w**2-2*A31*A41*wp**2-2*A32*A42*wp**2+2*A33*A43*wp**2
    +2*A34*A44*wp**2)*np.cos(2*t*w)+(A41**2+A42**2-A43**2-A44**2)*(w-wp)**2*np.cos(2*t*(w-wp))
    +2*A11*A41*w**2*np.cos(t*(2*w-wp))+2*A12*A42*w**2*np.cos(t*(2*w-wp))-2*A13*A43*w**2*np.cos(t*(2*w-wp))
    -2*A14*A44*w**2*np.cos(t*(2*w-wp))-2*A11*A41*w*wp*np.cos(t*(2*w-wp))-2*A12*A42*w*wp*np.cos(t*(2*w-wp))
    +2*A13*A43*w*wp*np.cos(t*(2*w-wp))+2*A14*A44*w*wp*np.cos(t*(2*w-wp))+2*A11*A31*w**2*np.cos(t*wp)
    +2*A12*A32*w**2*np.cos(t*wp)+2*A13*A33*w**2*np.cos(t*wp)+2*A14*A34*w**2*np.cos(t*wp)
    +2*A11*A41*w**2*np.cos(t*wp)+2*A12*A42*w**2*np.cos(t*wp)+2*A13*A43*w**2*np.cos(t*wp)
    +2*A14*A44*w**2*np.cos(t*wp)+2*A11*A31*w*wp*np.cos(t*wp)+2*A12*A32*w*wp*np.cos(t*wp)
    +2*A13*A33*w*wp*np.cos(t*wp)+2*A14*A34*w*wp*np.cos(t*wp)-2*A11*A41*w*wp*np.cos(t*wp)
    -2*A12*A42*w*wp*np.cos(t*wp)-2*A13*A43*w*wp*np.cos(t*wp)-2*A14*A44*w*wp*np.cos(t*wp)
    +2*A31*A41*w**2*np.cos(2*t*wp)+2*A32*A42*w**2*np.cos(2*t*wp)+2*A33*A43*w**2*np.cos(2*t*wp)
    +2*A34*A44*w**2*np.cos(2*t*wp)+2*A21**2*wp**2*np.cos(2*t*wp)-2*A22**2*wp**2*np.cos(2*t*wp)
    -2*A31*A41*wp**2*np.cos(2*t*wp)-2*A32*A42*wp**2*np.cos(2*t*wp)-2*A33*A43*wp**2*np.cos(2*t*wp)
    -2*A34*A44*wp**2*np.cos(2*t*wp)+A31**2*w**2*np.cos(2*t*(w+wp))+A32**2*w**2*np.cos(2*t*(w+wp))
    -A33**2*w**2*np.cos(2*t*(w+wp))-A34**2*w**2*np.cos(2*t*(w+wp))+2*A31**2*w*wp*np.cos(2*t*(w+wp))
    +2*A32**2*w*wp*np.cos(2*t*(w+wp))-2*A33**2*w*wp*np.cos(2*t*(w+wp))-2*A34**2*w*wp*np.cos(2*t*(w+wp))
    +A31**2*wp**2*np.cos(2*t*(w+wp))+A32**2*wp**2*np.cos(2*t*(w+wp))-A33**2*wp**2*np.cos(2*t*(w+wp))
    -A34**2*wp**2*np.cos(2*t*(w+wp))+2*A11*A31*w**2*np.cos(t*(2*w+wp))+2*A12*A32*w**2*np.cos(t*(2*w+wp))
    -2*A13*A33*w**2*np.cos(t*(2*w+wp))-2*A14*A34*w**2*np.cos(t*(2*w+wp))+2*A11*A31*w*wp*np.cos(t*(2*w+wp))
    +2*A12*A32*w*wp*np.cos(t*(2*w+wp))-2*A13*A33*w*wp*np.cos(t*(2*w+wp))-2*A14*A34*w*wp*np.cos(t*(2*w+wp))
    +2*A11*A13*w**2*np.sin(2*t*w)+2*A12*A14*w**2*np.sin(2*t*w)+2*A33*A41*w**2*np.sin(2*t*w)
    +2*A34*A42*w**2*np.sin(2*t*w)+2*A31*A43*w**2*np.sin(2*t*w)+2*A32*A44*w**2*np.sin(2*t*w)
    -2*A33*A41*wp**2*np.sin(2*t*w)-2*A34*A42*wp**2*np.sin(2*t*w)-2*A31*A43*wp**2*np.sin(2*t*w)
    -2*A32*A44*wp**2*np.sin(2*t*w)+2*A41*A43*w**2*np.sin(2*t*(w-wp))+2*A42*A44*w**2*np.sin(2*t*(w-wp))
    -4*A41*A43*w*wp*np.sin(2*t*(w-wp))-4*A42*A44*w*wp*np.sin(2*t*(w-wp))+2*A41*A43*wp**2*np.sin(2*t*(w-wp))
    +2*A42*A44*wp**2*np.sin(2*t*(w-wp))+2*A13*A41*w**2*np.sin(t*(2*w-wp))+2*A14*A42*w**2*np.sin(t*(2*w-wp))
    +2*A11*A43*w**2*np.sin(t*(2*w-wp))+2*A12*A44*w**2*np.sin(t*(2*w-wp))-2*A13*A41*w*wp*np.sin(t*(2*w-wp))
    -2*A14*A42*w*wp*np.sin(t*(2*w-wp))-2*A11*A43*w*wp*np.sin(t*(2*w-wp))-2*A12*A44*w*wp*np.sin(t*(2*w-wp))
    -2*A13*A31*w**2*np.sin(t*wp)-2*A14*A32*w**2*np.sin(t*wp)+2*A11*A33*w**2*np.sin(t*wp)+2*A12*A34*w**2*np.sin(t*wp)
    +2*A13*A41*w**2*np.sin(t*wp)+2*A14*A42*w**2*np.sin(t*wp)-2*A11*A43*w**2*np.sin(t*wp)-2*A12*A44*w**2*np.sin(t*wp)
    -2*A13*A31*w*wp*np.sin(t*wp)-2*A14*A32*w*wp*np.sin(t*wp)+2*A11*A33*w*wp*np.sin(t*wp)+2*A12*A34*w*wp*np.sin(t*wp)
    -2*A13*A41*w*wp*np.sin(t*wp)-2*A14*A42*w*wp*np.sin(t*wp)+2*A11*A43*w*wp*np.sin(t*wp)+2*A12*A44*w*wp*np.sin(t*wp)
    +2*A33*A41*w**2*np.sin(2*t*wp)+2*A34*A42*w**2*np.sin(2*t*wp)-2*A31*A43*w**2*np.sin(2*t*wp)
    -2*A32*A44*w**2*np.sin(2*t*wp)+4*A21*A22*wp**2*np.sin(2*t*wp)-2*A33*A41*wp**2*np.sin(2*t*wp)
    -2*A34*A42*wp**2*np.sin(2*t*wp)+2*A31*A43*wp**2*np.sin(2*t*wp)+2*A32*A44*wp**2*np.sin(2*t*wp)
    +2*A31*A33*w**2*np.sin(2*t*(w+wp))+2*A32*A34*w**2*np.sin(2*t*(w+wp))+4*A31*A33*w*wp*np.sin(2*t*(w+wp))
    +4*A32*A34*w*wp*np.sin(2*t*(w+wp))+2*A31*A33*wp**2*np.sin(2*t*(w+wp))+2*A32*A34*wp**2*np.sin(2*t*(w+wp))
    +2*A13*A31*w**2*np.sin(t*(2*w+wp))+2*A14*A32*w**2*np.sin(t*(2*w+wp))+2*A11*A33*w**2*np.sin(t*(2*w+wp))
    +2*A12*A34*w**2*np.sin(t*(2*w+wp))+2*A13*A31*w*wp*np.sin(t*(2*w+wp))+2*A14*A32*w*wp*np.sin(t*(2*w+wp))
    +2*A11*A33*w*wp*np.sin(t*(2*w+wp))+2*A12*A34*w*wp*np.sin(t*(2*w+wp)))

    
def generate_profile(mpl=1,teq=600, cmf=0.01, log_feh=0, co=0.55, tint=100, fixed_radius=None):
    """
    Uses GASTLI to generate cubic spline-interpolated temperature and pressure profile of a planet.
    
    mpl = planet mass (Jupiter masses)
    teq = equilibrium temperature of the planet (K)
    cmf = core mass fraction of planet
    log_feh = log metallicity of planet.
    co = C/O ratio of planet
    tint = internal temperature of planet (K)
    fixed_radius = Value that designates if we want to user-input the radius.  
       if fixed_radius==None, GASTLI will calculate the planet's radius and output that value.
       if fixed_radius==(float), GASTLI will adjust the model outputs to force the planet's radius to 
           that value.  Make sure it's in meters if you go this route.
    mpl = planet mass (Jupiter masses)
    teq = equilibrium temperature of the planet (K)
    cmf = core mass fraction of planet
    log_feh = log metallicity of planet.
    co = C/O ratio of planet
    tint = internal temperature of planet (K)    
    
    """
    # Generate a model with GASTLI.
    if mpl>=1:
        coupling = cpl.coupling(pow_law_formass=0.31)
    else: #As suggested by the code, for low-mass bodies, decrease the pow_law_formass parameter
        coupling = cpl.coupling(pow_law_formass = 0.3)

    coupling.main(mpl*MJUP/MEARTH, cmf, teq, tint, CO=co, log_FeH=log_feh)

    # Radius of planet, total.
    rtot = coupling.Rtot * RJUP

    # Make temperature-pressure profiles by pulling out the values from both the bulk + atmosphere models.

    # first, determine the radius limit of the bulk model.
    idx_upper = np.where(coupling.myplanet.r < np.min(coupling.myatmmodel.r))

    # Now, concatenate the arrays from the bulk+atmosphere models
    if fixed_radius==None:
    	rs = np.concatenate((coupling.myplanet.r[idx_upper][:-1], coupling.myatmmodel.r))
    elif type(fixed_radius)==float or type(fixed_radius)==np.float64:
    	rs = np.concatenate((coupling.myplanet.r[idx_upper][:-1], coupling.myatmmodel.r)) * fixed_radius/rtot
    	rtot = fixed_radius

    ts = np.concatenate((coupling.myplanet.T[idx_upper][:-1], coupling.myatmmodel.T_ode))
    ps = np.concatenate((coupling.myplanet.P[idx_upper][:-1], coupling.myatmmodel.P_ode))
    rhos = np.concatenate((coupling.myplanet.rho[idx_upper][:-1], coupling.myatmmodel.rho_ode))
    idxsort = np.argsort(rs)
    rs=rs[idxsort]
    ts=ts[idxsort]
    ps=ps[idxsort]
    rhos=rhos[idxsort]
    # interpolate these profiles using cubicsplines.
    tprofile = CubicSpline(rs, ts)
    pprofile = CubicSpline(rs, ps)
    rhoprofile = CubicSpline(rs, rhos)
    
    return rtot, tprofile, pprofile, rhoprofile
    
def find_power(
    ts,
    ecc=0,
    r0=0.0001,
    p=1,
    alpha=np.pi / 2,
    bsurf=4 * 10**3,
    mstar=1,
    rstar=1,
    tstar=5778,
    rotper=2,
    tsurf=None,
    temp_mult=1,
    eta_mult=1,
    eta_func="base",
    plot=False,
    use_GASTLI=False,
    fixed_radius=None,
    nr=1000,
    mp=1,
    cmf=0.01,
    log_feh=0,
    co=0.55,
    tint=100,
    l=1,
    pregen_tprofile=None,
    pregen_pprofile=None,
    pregen_rhoprofile=None,
    pregen_r = None,
):
    """
    ts are times to evaluate power, in seconds.
    ecc is planet eccentricity
    r0 is an arbitrary small reference radius (m)
    p is planet orbital period, in days
    alpha is the angle of the stellar magnetic field, in radians
    bsurf is the stellar surface magnetic field, in Gauss
    mstar is stellar mass, in Msolar
    rstar is stellar radius, in Rsolar
    tstar is stellar effective temperature, in K 
    rotper is stellar rotation period, in days
    tsurf allows us to manually define a planet surface temperature.  If left at default,
    the code calculates tsurf = teq.
    eta_mult = multiplicative factor on eta
    eta_func = which eta function to use.  "base" uses the laine formulation, "alkali"  uses the
    formulation that incorporates Na and K
    temp_mult = multiplicative factor on temperature
    plot = boolean, show plots if True
    use_GASTLI = boolean, use GASTLI to estimate the temperature-pressure profile if true (recommended)
    fixed_radius = Value that designates if we want to user-input the radius.  
       if fixed_radius==None, GASTLI will calculate the planet's radius and output that value.
       if fixed_radius==(float), GASTLI will adjust the model outputs to force the planet's radius to 
           that value.  Make sure it's in meters if you go this route.
    nr is resolution of radius grid over which to calculate Pvol 
    
    mpl = planet mass (Jupiter masses)
    teq = equilibrium temperature of the planet (K)
    cmf = core mass fraction of planet
    log_feh = log metallicity of planet.
    co = C/O ratio of planet
    tint = internal temperature of planet (K)   

    l = spherical harmonics to consider.  For nonzero eccentricity, only l=1 is allowed.  For zero eccentricity, 
    l=2 or l=1 are both valid.
    """
    

    # calculate magnetic moment.
    m = (rstar * RSOLAR) ** 3 * (bsurf / 10000) / 2 * 4 * np.pi / MU0
    # calculate the semi-major axis.
    a = ((p / 365) ** 2 * mstar) ** (1 / 3)
    d = a * AU

    # Calculate w.
    wp = 2 * np.pi / (p * 24 * 3600)
    wstar = 2 * np.pi / (rotper * 24 * 3600)
    w = wstar - wp


    teq = tstar * np.sqrt(rstar * RSOLAR / (a * AU)) * (1 / 4) ** (1 / 4)
    if use_GASTLI == True:
    
    	if pregen_tprofile==None and pregen_pprofile==None and pregen_r==None:
            rp, tprofile, pprofile, rhoprofile = generate_profile(
                mpl=mp, teq=teq, cmf=cmf, log_feh=log_feh, co=co, tint=tint, fixed_radius=fixed_radius,
            )
    	else:
            rp, tprofile, pprofile, rhoprofile = pregen_r, pregen_tprofile, pregen_pprofile, pregen_rhoprofile
        
    	if eta_func=="base":

        	def eta_use(r, rp, temp_mult=1, eta_mult=1):
                
                
                
                    eta= (
                        1.28
                        * 10**-2
                        * np.sqrt(pprofile(r))
                        / (temp_mult * tprofile(r)) ** (3 / 4)
                        * np.exp(78909 / (temp_mult * tprofile(r)))
                        ) * eta_mult
                
                    if len(eta[np.isnan(eta)]>0):
                        idx = np.arange(len(eta))[np.isnan(eta)]
                        print(r[idx], tprofile(r[idx]), pprofile(r[idx]))
                
                    return eta
                    
                    
    	elif eta_func=="alkali": #WARNING: this is extremely slow.
        
            def eta_use_helper(r, rp, temp_mult=temp_mult):
                    xk=4.339 # ionization energy of potassium, eV
                    xna=5.138 # ionization energy of sodium, eV
                    abk=-6.88 # log abundance of potassium
                    abna=-5.67 # log abundance of sodium
                    temp = tprofile(r) * temp_mult
                    nh = rhoprofile(r) / MH #TODO: density is also contributed to by helium, keep that in mind.  Also note that this is in general just a representation of 
                    # "the number density of every gas that is inert in terms of ionization at this temperature" and therefore includes both hydrogen and helium.
                    # Use sympy to solve the saha equation (TODO: find a way to do this in advance to speed it up, sympy is a litte wonky here)
                    nna0, nk0, nna1, nk1, a, b, nna, nk = sympy.symbols('N_{Na}^0, N_K^0, N_{Na}^1, N_{K}^1, a, b, "N_{K}", "N_{Na}')
                    # right-hand side of saha equation.
                    aval = 2 * 1/2 * (2*np.pi*ME*KB*temp / H**2)**(3/2) * np.e**(-xna/(8.617e-5*temp))
                    bval = 2 * 1/2 * (2*np.pi*ME*KB*temp / H**2)**(3/2) * np.e**(-xk/(8.617e-5*temp))
                    nhval=nh
                    nnaval = 10**abna*nhval
                    nkval = 10**abk*nhval
                    
                    
                    eq1 = sympy.Eq(nna1 *(nna1+nk1) / (nna - nna1),a)
                    nk1_new = sympy.solve(eq1, nk1)[0]
                    eq2 = sympy.simplify(sympy.Eq(nk1*(nna1+nk1) / (nk - nk1), b)).subs(nk1, nk1_new)#
                    eqn =sympy.simplify(eq2)
                    nna1_new = sympy.solve(eqn.subs(a, aval).subs(b, bval).subs(nna, nnaval).subs(nk, nkval), nna1)
                    solution = None
                    for i in nna1_new:
                        if i.as_real_imag()[0] > 0:
                            solution = i.as_real_imag()[0]
                    if solution ==None:
                        print("No solution for eta at T={temp:.0f}, rho={rho:.0f}".format(temp=temp, rho=rho))
                        return np.nan
                    else:
                        nna1_new = nna1_new[2].as_real_imag()[0]
                        nk1_new = nk1_new.subs(nna1, nna1_new).subs(nna, nnaval).subs(a, aval)
                        ne_new = nk1_new + nna1_new
                        nna0 = nnaval - nna1_new
                        nk0 = nkval - nk1_new
                        nn = nk0+nna0 + nhval

                        eta_calc = 0.023 * nn/ne_new * temp**(1/2)
                        
                        return eta_calc
                    
            r_profile = np.linspace(r0, rp, nr)
            etas=np.zeros(nr)
            for i in range(nr):
                    etas[i]=eta_use_helper(r_profile[i], rp)
            etaprofile=CubicSpline(r_profile, etas)
                    
            def eta_use(r, rp, temp_mult=1, eta_mult=1):
                    return eta_mult * etaprofile(r)
    	elif type(eta_func) == CubicSpline: #user-input profile.
            def eta_use(r, rp, temp_mult=1, eta_mult=1):
                    return eta_mult * eta_func(r)    	    

    else:
        eta_use=eta_simple
        rp=fixed_radius

    pinds = np.zeros(len(ts))  # induction power as a function of time.
    dts = np.zeros(len(ts))  # change in surface temp as a function of time.
    teqs = np.zeros(len(ts))  # equlibrium temperature as a function of time.
    r_profile = np.linspace(r0, rp, nr)

    y1 = find_ys(w, rp, eta_use, r0=r0, eta_mult=eta_mult, temp_mult=temp_mult)
    if ecc>0:
        y2 = find_ys(wp, rp, eta_use, r0=r0, eta_mult=eta_mult, temp_mult=temp_mult)
        y3 = find_ys(w + wp, rp, eta_use, r0=r0, eta_mult=eta_mult, temp_mult=temp_mult)
        y4 = find_ys(w - wp, rp, eta_use, r0=r0, eta_mult=eta_mult, temp_mult=temp_mult)

        coeffs = find_coeffs(rp, y1, y2, y3, y4, alpha, m, d, ecc)
        a_vals = find_as(r_profile, y1, y2, y3, y4, coeffs) 
    elif ecc==0:
        coeffs = find_coeffs_zeroecc(rp, y1, alpha, m, d)
        a_vals = find_as_zeroecc(r_profile, y1, coeffs)         

    planet = Planet(t0=0, m0=0, ecc=ecc, a=a, ms=mstar)


    pos, _ = planet.calc_pos(ts / 24 / 3600)  # planet function takes times in days.
    ds = np.sqrt(np.sum(pos**2, axis=0)) / 100  # the planet function
    
    pind_profile = np.zeros(len(r_profile))

    # Loop over and calculate the power at each time.
    for i in range(len(ts)):

        teq = tstar * np.sqrt(rstar * RSOLAR / ds[i]) * (1 / 4) ** (1 / 4)
        teqs[i] = teq
        if ecc>0:
            p_r = pvol(
                ts[i], w, wp, r_profile, rp, a_vals, eta_use, eta_mult=eta_mult, temp_mult=temp_mult
            )
        elif ecc==0:
            p_r = pvol_zeroecc(
                ts[i], w, r_profile, rp, a_vals, eta_use, eta_mult=eta_mult, temp_mult=temp_mult,l=l
            )

        # save the power profile for the first value in the ts array
        if i==0:
            pind_profile = p_r
        pheat = scipy.integrate.simpson(p_r * r_profile**2, r_profile)

        tsurf = (pheat / (4 * np.pi * SB * rp**2) + teq**4) ** (1 / 4)
        dts[i] = tsurf - teq
        pinds[i] = pheat

    return pinds, dts, teqs, r_profile, pind_profile, eta_use(r_profile, rp)


def find_power_lambda_zeroalpha(
    ts,
    lam=np.pi/2,
    r0=0.0001,
    p=1,
    bsurf=4 * 10**3,
    mstar=1,
    rstar=1,
    tstar=5778,
    tsurf=None,
    temp_mult=1,
    eta_mult=1,
    plot=False,
    use_GASTLI=False,
    fixed_radius=None,
    nr=1000,
    mp=1,
    cmf=0.01,
    log_feh=0,
    co=0.55,
    tint=100,
    pregen_tprofile=None,
    pregen_pprofile=None,
    pregen_rhoprofile=None,
    pregen_r = None,
):
    """
    Finds the power, similarly to find_power(), but in the scenario where
    alpha = ecc = 0.  
    lam = the obliquity of the star, in radians. 
    """
    

    # calculate magnetic moment.
    m = (rstar * RSOLAR) ** 3 * (bsurf / 10000) / 2 * 4 * np.pi / MU0
    # calculate the semi-major axis.
    a = ((p / 365) ** 2 * mstar) ** (1 / 3)
    d = a * AU

    # Calculate w.
    wp = 2 * np.pi / (p * 24 * 3600)

    teq = tstar * np.sqrt(rstar * RSOLAR / (a * AU)) * (1 / 4) ** (1 / 4)
    if use_GASTLI == True:
    
    	if pregen_tprofile==None and pregen_pprofile==None and pregen_r==None:
            rp, tprofile, pprofile, rhoprofile = generate_profile(
                mpl=mp, teq=teq, cmf=cmf, log_feh=log_feh, co=co, tint=tint, fixed_radius=fixed_radius,
            )
    	else:
            rp, tprofile, pprofile, rhoprofile = pregen_r, pregen_tprofile, pregen_pprofile, pregen_rhoprofile


    	def eta_use(r, rp, temp_mult=1, eta_mult=1):
            
            
            
            eta= (
                1.28
                * 10**-2
                * np.sqrt(pprofile(r))
                / (temp_mult * tprofile(r)) ** (3 / 4)
                * np.exp(78909 / (temp_mult * tprofile(r)))
            ) * eta_mult
            
            if len(eta[np.isnan(eta)]>0):
                idx = np.arange(len(eta))[np.isnan(eta)]
                print(r[idx], tprofile(r[idx]), pprofile(r[idx]))
            
            return eta
    else:
        eta_use=eta_simple
        rp=fixed_radius

    pinds = np.zeros(len(ts))  # induction power as a function of time.
    dts = np.zeros(len(ts))  # change in surface temp as a function of time.
    teqs = np.zeros(len(ts))  # equlibrium temperature as a function of time.
    r_profile = np.linspace(r0, rp, nr)

    # Finding the y functions identically to the zero-lambda case, but with w = wp.
    y = find_ys(wp, rp, eta_use, r0=r0, eta_mult=eta_mult, temp_mult=temp_mult)
    coeffs = find_coeffs_lam(rp, y, lam, m, d)
    a_vals = find_as_zeroecc(r_profile, y, coeffs) 

    planet = Planet(t0=0, m0=0, ecc=0, a=a, ms=mstar)

    pind_profile = np.zeros(len(r_profile))
    teq = tstar * np.sqrt(rstar * RSOLAR / d) * (1 / 4) ** (1 / 4)

    # Loop over and calculate the power at each time.
    for i in range(len(ts)):

        
        teqs[i] = teq

        p_r = pvol_zeroecc( # Evaluate the pvol, but with w=wp.
            ts[i],wp, r_profile, rp, a_vals, eta_use, eta_mult=eta_mult, temp_mult=temp_mult, l=1
        )
        

        # save the power profile for the first value in the ts array
        if i==0:
            pind_profile = p_r
        pheat = scipy.integrate.simpson(p_r * r_profile**2, r_profile)

        tsurf = (pheat / (4 * np.pi * SB * rp**2) + teq**4) ** (1 / 4)
        dts[i] = tsurf - teq
        pinds[i] = pheat

    return pinds, dts, teqs, r_profile, pind_profile, eta_use(r_profile, rp)



def find_as_lam(r, y1s, y2s, y5s, coeffs):
    """
    find the A coefficients as defined in our paper.
    inputs:
    rp = radius of planet (m)
    y1s, y2s, y5s  = outputs of find_ys for the i=1, i=2, i=5 cases
    coeffs = output of find_coeffs for this model.
    
    """
    reg11 = y1s["y11"](r)
    img11 = y1s["y12"](r)
    reg21 = y2s["y11"](r)
    img21 = y2s["y12"](r)
    reg51 = y4s["y11"](r)
    img51 = y4s["y12"](r)


    a_vals = {}

    a_vals["a11"] = (-coeffs["nu111"] + coeffs["nu11m1"]) * reg11 + (
        -coeffs["mu111"] + coeffs["mu11m1"]
    ) * img11
    a_vals["a12"] = (-coeffs["mu111"] - coeffs["mu11m1"]) * reg11 + (
        coeffs["nu111"] + coeffs["nu11m1"]
    ) * img11
    a_vals["a13"] = (-coeffs["mu111"] + coeffs["mu11m1"]) * reg11 + (
        coeffs["nu111"] - coeffs["nu11m1"]
    ) * img11
    a_vals["a14"] = (coeffs["nu111"] + coeffs["nu11m1"]) * reg11 + (
        coeffs["mu111"] + coeffs["mu11m1"]
    ) * img11
    
    a_vals["a21"] = (-coeffs["nu211"] + coeffs["nu21m1"]) * reg21 + (
        -coeffs["mu211"] + coeffs["mu21m1"]
    ) * img21
    a_vals["a22"] = (-coeffs["mu211"] - coeffs["mu21m1"]) * reg21 + (
        coeffs["nu211"] + coeffs["nu21m1"]
    ) * img21
    a_vals["a23"] = (-coeffs["mu211"] + coeffs["mu21m1"]) * reg21 + (
        coeffs["nu211"] - coeffs["nu21m1"]
    ) * img21
    a_vals["a24"] = (coeffs["nu211"] + coeffs["nu21m1"]) * reg21 + (
        coeffs["mu211"] + coeffs["mu21m1"]
    ) * img21
    a_vals["a25"] = coeffs["nu210"] * reg21 + coeffs["mu210"] * img21
    a_vals["a26"] = coeffs["mu210"] * reg21 - coeffs["nu210"] * img21

    a_vals["a51"] = (-coeffs["nu511"] + coeffs["nu51m1"]) * reg51 + (
        -coeffs["mu511"] + coeffs["mu51m1"]
    ) * img51
    a_vals["a52"] = (-coeffs["mu511"] - coeffs["mu51m1"]) * reg51 + (
        coeffs["nu511"] + coeffs["nu51m1"]
    ) * img51
    a_vals["a53"] = (-coeffs["mu511"] + coeffs["mu51m1"]) * reg51 + (
        coeffs["nu511"] - coeffs["nu51m1"]
    ) * img51
    a_vals["a54"] = (coeffs["nu511"] + coeffs["nu51m1"]) * reg51 + (
        coeffs["mu511"] + coeffs["mu51m1"]
    ) * img51

    return a_vals
    
    
def find_power_lam(
    ts,
    r0=0.0001,
    p=1,
    alpha=np.pi / 2,
    lam=0,
    bsurf=4 * 10**3,
    mstar=1,
    rstar=1,
    tstar=5778,
    rotper=2,
    tsurf=None,
    temp_mult=1,
    eta_mult=1,
    plot=False,
    use_GASTLI=False,
    fixed_radius=None,
    nr=1000,
    mp=1,
    cmf=0.01,
    log_feh=0,
    co=0.55,
    tint=100,
    l=1,
    pregen_tprofile=None,
    pregen_pprofile=None,
    pregen_rhoprofile=None,
    pregen_r = None,
):
    """
    ts are times to evaluate power, in seconds.
    r0 is an arbitrary small reference radius (m)
    p is planet orbital period, in days
    alpha is the angle of the stellar magnetic field, in radians
    lambda is the obliquity, in radians
    bsurf is the stellar surface magnetic field, in Gauss
    mstar is stellar mass, in Msolar
    rstar is stellar radius, in Rsolar
    tstar is stellar effective temperature, in K 
    rotper is stellar rotation period, in days
    tsurf allows us to manually define a planet surface temperature.  If left at default,
    the code calculates tsurf = teq.
    eta_mult = multiplicative factor on eta
    temp_mult = multiplicative factor on temperature
    plot = boolean, show plots if True
    use_GASTLI = boolean, use GASTLI to estimate the temperature-pressure profile if true (recommended)
    fixed_radius = Value that designates if we want to user-input the radius.  
       if fixed_radius==None, GASTLI will calculate the planet's radius and output that value.
       if fixed_radius==(float), GASTLI will adjust the model outputs to force the planet's radius to 
           that value.  Make sure it's in meters if you go this route.
    nr is resolution of radius grid over which to calculate Pvol 
    
    mpl = planet mass (Jupiter masses)
    teq = equilibrium temperature of the planet (K)
    cmf = core mass fraction of planet
    log_feh = log metallicity of planet.
    co = C/O ratio of planet
    tint = internal temperature of planet (K)   

    l = spherical harmonics to consider.  For nonzero eccentricity, only l=1 is allowed.  For zero eccentricity, 
    l=2 or l=1 are both valid.
    """
    

    # calculate magnetic moment.
    m = (rstar * RSOLAR) ** 3 * (bsurf / 10000) / 2 * 4 * np.pi / MU0
    # calculate the semi-major axis.
    a = ((p / 365) ** 2 * mstar) ** (1 / 3)
    d = a * AU

    # Calculate w.
    wp = 2 * np.pi / (p * 24 * 3600)
    wstar = 2 * np.pi / (rotper * 24 * 3600)
    w = wstar - wp


    teq = tstar * np.sqrt(rstar * RSOLAR / (a * AU)) * (1 / 4) ** (1 / 4)
    if use_GASTLI == True:
    
    	if pregen_tprofile==None and pregen_pprofile==None and pregen_r==None:
            rp, tprofile, pprofile, rhoprofile = generate_profile(
                mpl=mp, teq=teq, cmf=cmf, log_feh=log_feh, co=co, tint=tint, fixed_radius=fixed_radius,
            )
    	else:
            rp, tprofile, pprofile, pregen_rhoprofile = pregen_r, pregen_tprofile, pregen_pprofile, pregen_rhoprofile


    	def eta_use(r, rp, temp_mult=1, eta_mult=1):
            
            
            
            eta= (
                1.28
                * 10**-2
                * np.sqrt(pprofile(r))
                / (temp_mult * tprofile(r)) ** (3 / 4)
                * np.exp(78909 / (temp_mult * tprofile(r)))
            ) * eta_mult
            
            if len(eta[np.isnan(eta)]>0):
                idx = np.arange(len(eta))[np.isnan(eta)]
                print(r[idx], tprofile(r[idx]), pprofile(r[idx]))
            
            return eta
    else:
        eta_use=eta_simple
        rp=fixed_radius

    pinds = np.zeros(len(ts))  # induction power as a function of time.
    dts = np.zeros(len(ts))  # change in surface temp as a function of time.
    teqs = np.zeros(len(ts))  # equlibrium temperature as a function of time.
    r_profile = np.linspace(r0, rp, nr)

    y1 = find_ys(w, rp, eta_use, r0=r0, eta_mult=eta_mult, temp_mult=temp_mult)
    y2 = find_ys(wp, rp, eta_use, r0=r0, eta_mult=eta_mult, temp_mult=temp_mult)
    y5 = find_ys(w + 2*wp, rp, eta_use, r0=r0, eta_mult=eta_mult, temp_mult=temp_mult)

    coeffs = find_coeffs_lam_alpha(rp, y1, y2, y5, lam, alpha, m, d)
    a_vals = find_as_lam(r_profile, y1, y2, y5, coeffs) 

    planet = Planet(t0=0, m0=0, ecc=ecc, a=a, ms=mstar)

    pos, _ = planet.calc_pos(ts / 24 / 3600)  # planet function takes times in days.
    ds = np.sqrt(np.sum(pos**2, axis=0)) / 100  # the planet function
    
    pind_profile = np.zeros(len(r_profile))

    # Loop over and calculate the power at each time.
    for i in range(len(ts)):

        teq = tstar * np.sqrt(rstar * RSOLAR / ds[i]) * (1 / 4) ** (1 / 4)
        teqs[i] = teq
        if ecc>0:
            p_r = pvol(
                ts[i], w, wp, r_profile, rp, a_vals, eta_use, eta_mult=eta_mult, temp_mult=temp_mult
            )
        elif ecc==0:
            p_r = pvol_zeroecc(
                ts[i], w, r_profile, rp, a_vals, eta_use, eta_mult=eta_mult, temp_mult=temp_mult,l=1
            )

        # save the power profile for the first value in the ts array
        if i==0:
            pind_profile = p_r
        pheat = scipy.integrate.simpson(p_r * r_profile**2, r_profile)

        tsurf = (pheat / (4 * np.pi * SB * rp**2) + teq**4) ** (1 / 4)
        dts[i] = tsurf - teq
        pinds[i] = pheat

    return pinds, dts, teqs, r_profile, pind_profile, eta_use(r_profile, rp)

