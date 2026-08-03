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



def find_coeffs(rp, y1s, y2s, y3s, alpha, m, a, e):
    """find the mu and nu coefficients defined in Laine+ 2008.
    
    inputs:
    rp = radius of planet (m)
    y1s, y2s, y3s = outputs of find_ys for the i=1, i=2, and i=3 cases as defined in 
    section 3.4.2 of the paper.
    alpha = inclination angle (radians)
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
            [reg21, -img21, -1 / rp * np.sqrt(4 * np.pi / 3) , 0],
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

    # Now, the cross terms...

    reg31 = y3s["y11"](rp)
    img31 = y3s["y12"](rp)
    reg31dot = y3s["y13"](rp)
    img31dot = y3s["y14"](rp)

    # mu311, mu31m1, nu311, nu31m1, xi1, xi2, xi3, xi4
    eqns = np.array(
        [
            [img31, -img31, reg31, -reg31, -np.sqrt(8 * np.pi / 3) * 1 / rp, 0, 0, 0],
            [-reg31, reg31, img31, -img31, 0, -np.sqrt(8 * np.pi / 3) * 1 / rp, 0, 0],
            [reg31, reg31, -img31, -img31, 0, 0, -np.sqrt(8 * np.pi / 3) * 1 / rp, 0],
            [img31, img31, reg31,   reg31, 0, 0, 0, -np.sqrt(8 * np.pi / 3) * 1 / rp],
            [img31dot, -img31dot, reg31dot, -reg31dot, np.sqrt(8 * np.pi / 3) * 1 / rp**2, 0, 0, 0,],
            [-reg31dot, reg31dot, img31dot, -img31dot, 0, np.sqrt(8 * np.pi / 3) * 1 / rp**2, 0, 0,],
            [reg31dot, reg31dot, -img31dot, -img31dot, 0, 0, np.sqrt(8 * np.pi / 3) * 1 / rp**2, 0,],
            [img31dot, img31dot,  reg31dot,  reg31dot, 0, 0, 0, np.sqrt(8 * np.pi / 3) * 1 / rp**2,],
        ]
    )

    sols = np.array(
        [
            -np.sqrt(8 * np.pi / 3) * 1 * m * rp**2 * np.sin(alpha) / (8 * np.pi * a**3) * 6 * e,
            0,
            0,
            -np.sqrt(8 * np.pi / 3) * 1 * m * rp**2 * np.sin(alpha) / (8 * np.pi * a**3) * 3 * e,
            -2 * np.sqrt(8 * np.pi / 3) * 1 * m * rp * np.sin(alpha) / (8 * np.pi * a**3) * 6 * e,
            0,
            0,
            -2 * np.sqrt(8 * np.pi / 3) * 1 * m * rp * np.sin(alpha) / (8 * np.pi * a**3) * 3 * e,
        ]
    )

    x = np.linalg.solve(eqns, sols)

    data["mu311"] = x[0]
    data["mu31m1"] = x[1]
    data["nu311"] = x[2]
    data["nu31m1"] = x[3]

    # l=2, m=0 case, wp terms.
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

    # l=2, m=0 case. wp w terms.
    reg32 = y3s["y21"](rp)
    img32 = y3s["y22"](rp)
    reg32dot = y3s["y23"](rp)
    img32dot = y3s["y24"](rp)

    # mu320, nu320, zeta1, zeta2
    eqns = np.array(
        [
            [-img31, -reg32, -np.sqrt(4 * np.pi / 5) * 1 / rp**2, 0],
            [reg32, -img32, 0, -np.sqrt(4 * np.pi / 5) * 1 / rp**2],
            [-img31dot, -reg32dot, 2 * np.sqrt(4 * np.pi / 5) * 1 / rp**3, 0],
            [reg32dot, -img32dot, 0, 2 * np.sqrt(4 * np.pi / 5) * 1 / rp**3],
        ]
    )

    sols = np.array(
        [
            4 * e * np.sqrt(4 * np.pi / 5) * m * np.sin(alpha) * rp**3 / 8 / np.pi / a**4,
            0,
            12 * e * np.sqrt(4 * np.pi / 5) * m * np.sin(alpha) * rp**2 / 8 / np.pi / a**4,
            0,
        ]
    )

    x = np.linalg.solve(eqns, sols)

    data["mu320"] = x[0]
    data["nu320"] = x[1]

    # l = 2 m=1 case.
    reg22 = y2s["y21"](rp)
    img22 = y2s["y22"](rp)
    reg22dot = y2s["y23"](rp)
    img22dot = y2s["y24"](rp)

    # mu221, mu22m1, nu221, nu22m1, Om1, Om2, Om3, Om4
    eqns = np.array(
        [
            [ img22, -img22, reg22, -reg22, -np.sqrt(24 * np.pi / 5) * 1 / rp**2, 0, 0, 0,],
            [ -reg22, reg22, img22, -img22, 0, -np.sqrt(24 * np.pi / 5) * 1 / rp**2, 0, 0,],
            [ reg22, reg22, -img22, -img22, 0, 0, -np.sqrt(24 * np.pi / 5) * 1 / rp**2, 0,],
            [img22, img22, reg22, reg22,    0, 0, 0, -np.sqrt(24 * np.pi / 5) * 1 / rp**2],
            [ img22dot, -img22dot, reg22dot, -reg22dot, 2 * np.sqrt(24 * np.pi / 5) * 1 / rp**3, 0, 0, 0,],
            [ -reg22dot, reg22dot, img22dot, -img22dot, 0, 2 * np.sqrt(24 * np.pi / 5) * 1 / rp**3, 0, 0,],
            [ reg22dot, reg22dot, -img22dot, -img22dot, 0, 0, 2 * np.sqrt(24 * np.pi / 5) * 1 / rp**3, 0,],
            [ img22dot, img22dot,  reg22dot,  reg22dot, 0, 0, 0, 2 * np.sqrt(24 * np.pi / 5) * 1 / rp**3,],
        ]
    )

    sols = np.array(
        [
            0,
            -np.sqrt(24 * np.pi / 5) * m * np.cos(alpha) * rp**3 / 12 / np.pi / a**4 * 4 * e,
            0,
            0,
            0,
            -3 * np.sqrt(24 * np.pi / 5) * m * np.cos(alpha) * rp**2 / 12 / np.pi / a**4 * 4 * e,
            0,
            0,
        ]
    )

    x = np.linalg.solve(eqns, sols)

    data["mu221"] = x[0]
    data["mu22m1"] = x[1]
    data["nu221"] = x[2]
    data["nu22m1"] = x[3]

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

    # cross terms

    # mu322, nu32m2, nu322, nu32m2, d1, d2, d3, d4
    eqns = np.array(
        [
            [-img32,-img32,-reg32,-reg32, -4 * np.sqrt(6 * np.pi / 5) * 1 / rp**2, 0, 0, 0,],
            [ reg32, reg32,-img32,-img32, 0, -4 * np.sqrt(6 * np.pi / 5) * 1 / rp**2, 0, 0,],
            [-reg32, reg32, img32,-img32, 0, 0, -4 * np.sqrt(6 * np.pi / 5) * 1 / rp**2, 0,],
            [-img32, img32,-reg32, reg32, 0, 0, 0, -4 * np.sqrt(6 * np.pi / 5) * 1 / rp**2,],
            [-img32dot,-img32dot,-reg32dot,-reg32dot, 4 * 2 * np.sqrt(6 * np.pi / 5) * 1 / rp**3, 0, 0, 0,],
            [ reg32dot, reg32dot,-img32dot,-img32dot, 0, 4 * 2 * np.sqrt(6 * np.pi / 5) * 1 / rp**3, 0, 0,],
            [-reg32dot, reg32dot, img32dot,-img32dot, 0, 0, 4 * 2 * np.sqrt(6 * np.pi / 5) * 1 / rp**3, 0,],
            [-img32dot, img32dot,-reg32dot, reg32dot, 0, 0, 0, 4 * 2 * np.sqrt(6 * np.pi / 5) * 1 / rp**3,],
        ]
    )

    sols = np.array(
        [
            -4 * np.sqrt(6 * np.pi / 5) * m * np.sin(alpha) * rp**3 * 2 * e / 8 / np.pi / a**4,
            0,
            0,
            -4 * np.sqrt(6 * np.pi / 5) * m * np.sin(alpha) * rp**3 * 4 / 3 * e / 8 / np.pi / a**4,
            -3 * 4 * np.sqrt(6 * np.pi / 5) * m * np.sin(alpha) * rp**2 * 2 * e / 8 / np.pi / a**4,
            0,
            0,
            -3 * 4 * np.sqrt(6 * np.pi / 5) * m * np.sin(alpha) * rp**2 * 4 / 3 * e / 8 / np.pi / a**4,
        ]
    )

    x = np.linalg.solve(eqns, sols)

    data["mu322"] = x[0]
    data["mu32m2"] = x[1]
    data["nu322"] = x[2]
    data["nu32m2"] = x[3]

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
            np.sqrt(8 * np.pi / 3) * 2 * m * np.sin(lam) * rp**2 / (8 * np.pi * a**3) * 1,
            0,
            np.sqrt(8 * np.pi / 3) * m * np.sin(lam) * rp**2 / (8 * np.pi * a**3) * 1,
            0,
            np.sqrt(8 * np.pi / 3) * 4 * m * np.sin(lam) * rp / (8 * np.pi * a**3) * 1,
            0,
            np.sqrt(8 * np.pi / 3) * 2 * m * np.sin(lam) * rp / (8 * np.pi * a**3 * 1),
            0,
        ]
    )

    x = np.linalg.solve(eqns, sols)

    data["mu11"] = x[0]
    data["mu1m1"] = x[1]
    data["nu11"] = x[2]
    data["nu1m1"] = x[3]

    # l=2, m=0 case, wp terms.
    reg2 = ys["y21"](rp)
    img2 = ys["y22"](rp)
    reg2dot = ys["y23"](rp)
    img2dot = ys["y24"](rp)

    # mu120, nu120, beta1, beta2
    eqns = np.array(
        [
            [-img2, -reg2, -np.sqrt(4 * np.pi / 5) * (1 / rp**2), 0],
            [reg2, -img2, 0, -np.sqrt(4 * np.pi / 5) * (1 / rp**2)],
            [-img2dot, -reg2dot, 2 * np.sqrt(4 * np.pi / 5) * (1 / rp**3), 0],
            [reg2dot, -img2dot, 0, 2 * np.sqrt(4 * np.pi / 5) * (1 / rp**3)],
        ]
    )

    sols = np.array(
        [
            np.sqrt(4 * np.pi / 5) * m * np.sin(lam) * rp**3 / 8 / np.pi / a**4,
            0,
            3 * np.sqrt(4 * np.pi / 5) * m * np.sin(lam) * rp**2 / 8 / np.pi / a**4,
            0,
        ]
    )

    x = np.linalg.solve(eqns, sols)

    data["mu20"] = x[0]
    data["nu20"] = x[1]



    # mu122, mu12m2, nu122, nu12m2, b1, b2, b3, b4
    eqns = np.array(
        [
            [-img2, -img2, -reg2, -reg2, -4 * np.sqrt(6 * np.pi / 5) * 1 / rp**2, 0, 0, 0,],
            [ reg2,  reg2, -img2, -img2, 0, -4 * np.sqrt(6 * np.pi / 5) * 1 / rp**2, 0, 0,],
            [-reg2,  reg2,  img2, -img2, 0, 0, -4 * np.sqrt(6 * np.pi / 5) * 1 / rp**2, 0,],
            [-img2,  img2, -reg2,  reg2, 0, 0, 0, -4 * np.sqrt(6 * np.pi / 5) * 1 / rp**2,],
            [-img2dot,-img2dot,-reg2dot,-reg2dot, 4 * 2 * np.sqrt(6 * np.pi / 5) * 1 / rp**3, 0, 0, 0,],
            [ reg2dot, reg2dot,-img2dot,-img2dot, 0, 4 * 2 * np.sqrt(6 * np.pi / 5) * 1 / rp**3, 0, 0,],
            [-reg2dot, reg2dot, img2dot,-img2dot, 0, 0, 4 * 2 * np.sqrt(6 * np.pi / 5) * 1 / rp**3, 0,],
            [-img2dot, img2dot,-reg2dot, reg2dot, 0, 0, 0, 4 * 2 ** np.sqrt(6 * np.pi / 5) * 1 / rp**3,],
        ]
    )
    sols = np.array(
        [
            -2 * np.sqrt(6 * np.pi / 5) * m * np.sin(lam) * rp**3 / 8 / np.pi / a**4,
            0,
            -4 / 3 * np.sqrt(6 * np.pi / 5) * m * np.sin(lam) * rp**3 / 8 / np.pi / a**4,
            0,
            -3 * 2 * np.sqrt(6 * np.pi / 5) * m * np.sin(lam) * rp**2 / 8 / np.pi / a**4,
            0,
            -3 * 4 / 3 * np.sqrt(6 * np.pi / 5) * m * np.sin(lam) * rp**2 / 8 / np.pi / a**4,
            0,
        ]
    )
    x = np.linalg.solve(eqns, sols)

    data["mu22"] = x[0]
    data["mu2m2"] = x[1]
    data["nu22"] = x[2]
    data["nu2m2"] = x[3]



    return data

def find_as(r, y1s, y2s, y3s, coeffs):
    """
    find the A coefficients as defined in our paper.
    inputs:
    rp = radius of planet (m)
    y1s, y2s, y3s = outputs of find_ys for the i=1, i=2, and i=3 cases as defined in 
    section 3.4.2 of the paper.
    coeffs = output of find_coeffs for this model.
    
    """
    reg11 = y1s["y11"](r)
    img11 = y1s["y12"](r)
    reg21 = y2s["y11"](r)
    img21 = y2s["y12"](r)
    reg31 = y3s["y11"](r)
    img31 = y3s["y12"](r)

    reg12 = y1s["y21"](r)
    img12 = y1s["y22"](r)
    reg22 = y2s["y21"](r)
    img22 = y2s["y22"](r)
    reg32 = y3s["y21"](r)
    img32 = y3s["y22"](r)

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

    a_vals["a21"] = coeffs["nu210"] * reg21 + coeffs["mu210"] * img21
    a_vals["a22"] = coeffs["mu210"] * reg21 - coeffs["nu210"] * img21
    
    a_vals["a23"] = reg22 * (-coeffs["nu221"] + coeffs["nu22m1"]) + img22 * (
        -coeffs["mu221"] + coeffs["mu22m1"]
    )
    a_vals["a24"] = reg22 * (-coeffs["mu221"] - coeffs["mu22m1"]) + img22 * (
        coeffs["nu221"] + coeffs["nu22m1"]
    )
    a_vals["a25"] = reg22 * (-coeffs["mu221"] + coeffs["mu22m1"]) + img22 * (
        coeffs["nu221"] - coeffs["nu22m1"]
    )
    a_vals["a26"] = reg22 * (coeffs["nu221"] + coeffs["nu22m1"]) + img22 * (
        coeffs["mu221"] + coeffs["mu22m1"]
    )

    a_vals["a31"] = (-coeffs["nu311"] + coeffs["nu31m1"]) * reg31 + (
        -coeffs["mu311"] + coeffs["mu31m1"]
    ) * img31
    a_vals["a32"] = (-coeffs["mu311"] - coeffs["mu31m1"]) * reg31 + (
        coeffs["nu311"] + coeffs["nu31m1"]
    ) * img31
    a_vals["a33"] = (-coeffs["mu311"] + coeffs["mu31m1"]) * reg31 + (
        coeffs["nu311"] - coeffs["nu31m1"]
    ) * img31
    a_vals["a34"] = (coeffs["nu311"] + coeffs["nu31m1"]) * reg31 + (
        coeffs["mu311"] + coeffs["mu31m1"]
    ) * img31

    a_vals["a35"] = reg32 * coeffs["nu320"] + img32 * coeffs["mu320"]
    a_vals["a36"] = reg32 * coeffs["mu320"] - img32 * coeffs["nu320"]
    
    a_vals["a37"] = reg32 * (coeffs["nu322"] + coeffs["nu32m2"]) + img32 * (
        coeffs["mu322"] + coeffs["mu32m2"]
    )
    a_vals["a38"] = reg32 * (-coeffs["nu322"] + coeffs["nu32m2"]) + img32 * (
        -coeffs["mu322"] + coeffs["mu32m2"]
    )
    a_vals["a39"] = reg32 * (coeffs["mu322"] - coeffs["mu32m2"]) + img32 * (
        -coeffs["nu322"] + coeffs["nu32m2"]
    )
    a_vals["a310"] = reg32 * (coeffs["mu322"] + coeffs["mu32m2"]) + img32 * (
        -coeffs["nu322"] - coeffs["nu32m2"]
    )

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

    a_vals["a11"] = (-coeffs["nu11"] + coeffs["nu1m1"]) * reg11 + (
        -coeffs["mu11"] + coeffs["mu1m1"]
    ) * img11
    a_vals["a12"] = (-coeffs["mu11"] - coeffs["mu1m1"]) * reg11 + (
        coeffs["nu11"] + coeffs["nu1m1"]
    ) * img11
    a_vals["a13"] = (-coeffs["mu11"] + coeffs["mu1m1"]) * reg11 + (
        coeffs["nu11"] - coeffs["nu1m1"]
    ) * img11
    a_vals["a14"] = (coeffs["nu11"] + coeffs["nu1m1"]) * reg11 + (
        coeffs["mu11"] + coeffs["mu1m1"]
    ) * img11

    a_vals["a15"] = reg12 * coeffs["nu20"] + img12 * coeffs["mu20"]
    a_vals["a16"] = reg12 * coeffs["mu20"] - img12 * coeffs["nu20"]

    a_vals["a17"] = reg12 * (coeffs["nu22"] + coeffs["nu2m2"]) + img12 * (
        coeffs["mu22"] + coeffs["mu2m2"]
    )
    a_vals["a18"] = reg12 * (coeffs["mu22"] - coeffs["mu2m2"]) + img12 * (
        -coeffs["nu22"] + coeffs["nu2m2"]
    )
    a_vals["a19"] = reg12 * (coeffs["mu22"] + coeffs["mu2m2"]) + img12 * (
        -coeffs["nu22"] - coeffs["nu2m2"]
    )
    a_vals["a110"] = reg12 * (-coeffs["nu22"] + coeffs["nu2m2"]) + img12 * (
        -coeffs["mu22"] + coeffs["mu2m2"]
    )
    
    # Set all of the cross terms equal to zero, as they are in the e=0 scenario.

    a_vals["a21"] = 0
    a_vals["a22"] = 0
    a_vals["a23"] = 0
    a_vals["a24"] = 0
    a_vals["a25"] = 0
    a_vals["a26"] = 0
    a_vals["a31"] = 0
    a_vals["a32"] = 0
    a_vals["a33"] = 0
    a_vals["a34"] = 0
    a_vals["a35"] = 0
    a_vals["a36"] = 0
    a_vals["a37"] = 0
    a_vals["a38"] = 0
    a_vals["a39"] = 0
    a_vals["a310"] = 0
    return a_vals
    
    
    
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
    A15=a_vals["a15"]
    A16=a_vals["a16"]
    A17=a_vals["a17"]
    A18=a_vals["a18"]
    A19=a_vals["a19"]
    A110=a_vals["a110"]
    A21=a_vals["a21"]
    A22=a_vals["a22"]
    A23=a_vals["a23"]
    A24=a_vals["a24"]
    A25=a_vals["a25"]
    A26=a_vals["a26"]
    A31=a_vals["a31"]
    A32=a_vals["a32"]
    A33=a_vals["a33"]
    A34=a_vals["a34"]
    A35=a_vals["a35"]
    A36=a_vals["a36"]
    A37=a_vals["a37"]
    A38=a_vals["a38"]
    A39=a_vals["a39"]
    A310=a_vals["a310"]
 
    return (1/(64*eta_func(r, rp, eta_mult=eta_mult, temp_mult=temp_mult)))/r**2*MU0*(32*A11**2*w**2+96*A110**2*w**2+32*A12**2*w**2+32*A13**2*w**2+32*A14**2*w**2
    +192*A15**2*w**2+192*A16**2*w**2+96*A17**2*w**2+96*A18**2*w**2+96*A19**2*w**2
    +32*A31**2*w**2+36*A310**2*w**2+32*A32**2*w**2+32*A33**2*w**2+32*A34**2*w**2
    +192*A35**2*w**2+192*A36**2*w**2+36*A37**2*w**2+36*A38**2*w**2+36*A39**2*w**2
    +9*np.sqrt(5)*A310*A33*np.pi*w**2+9*np.sqrt(5)*A31*A37*np.pi*w**2+9*np.sqrt(5)*A34*A38*np.pi*w**2
    +9*np.sqrt(5)*A32*A39*np.pi*w**2+64*A31**2*w*wp+72*A310**2*w*wp+64*A32**2*w*wp
    +64*A33**2*w*wp+64*A34**2*w*wp+384*A35**2*w*wp+384*A36**2*w*wp+72*A37**2*w*wp
    +72*A38**2*w*wp+72*A39**2*w*wp+18*np.sqrt(5)*A310*A33*np.pi*w*wp+18*np.sqrt(5)*A31*A37*np.pi*w*wp
    +18*np.sqrt(5)*A34*A38*np.pi*w*wp+18*np.sqrt(5)*A32*A39*np.pi*w*wp+64*A21**2*wp**2+64*A22**2*wp**2
    +96*A23**2*wp**2+96*A24**2*wp**2+96*A25**2*wp**2+96*A26**2*wp**2+32*A31**2*wp**2+36*A310**2*wp**2
    +32*A32**2*wp**2+32*A33**2*wp**2+32*A34**2*wp**2+192*A35**2*wp**2+192*A36**2*wp**2+36*A37**2*wp**2
    +36*A38**2*wp**2+36*A39**2*wp**2+9*np.sqrt(5)*A310*A33*np.pi*wp**2+9*np.sqrt(5)*A31*A37*np.pi*wp**2
    +9*np.sqrt(5)*A34*A38*np.pi*wp**2+9*np.sqrt(5)*A32*A39*np.pi*wp**2
    +32*(A11**2-3*A110**2+A12**2-A13**2-A14**2+6*A15**2-6*A16**2+3*A17**2+3*A18**2-3*A19**2)*w**2*np.cos(2*t*w)
    +(64*A11*A31+64*A12*A32+64*A13*A33+64*A14*A34+384*A15*A35+384*A16*A36+9*np.sqrt(5)*A13*A310*np.pi
    +9*np.sqrt(5)*A11*A37*np.pi+9*np.sqrt(5)*A14*A38*np.pi+9*np.sqrt(5)*A12*A39*np.pi)*w*(w+wp)*np.cos(t*wp)
    +64*A21**2*wp**2*np.cos(2*t*wp)-64*A22**2*wp**2*np.cos(2*t*wp)+96*A23**2*wp**2*np.cos(2*t*wp)
    +96*A24**2*wp**2*np.cos(2*t*wp)-96*A25**2*wp**2*np.cos(2*t*wp)-96*A26**2*wp**2*np.cos(2*t*wp)
    +32*A31**2*w**2*np.cos(2*t*(w+wp))-36*A310**2*w**2*np.cos(2*t*(w+wp))+32*A32**2*w**2*np.cos(2*t*(w+wp))
    -32*A33**2*w**2*np.cos(2*t*(w+wp))-32*A34**2*w**2*np.cos(2*t*(w+wp))+192*A35**2*w**2*np.cos(2*t*(w+wp))
    -192*A36**2*w**2*np.cos(2*t*(w+wp))+36*A37**2*w**2*np.cos(2*t*(w+wp))-36*A38**2*w**2*np.cos(2*t*(w+wp))
    +36*A39**2*w**2*np.cos(2*t*(w+wp))-9*np.sqrt(5)*A310*A33*np.pi*w**2*np.cos(2*t*(w+wp))
    +9*np.sqrt(5)*A31*A37*np.pi*w**2*np.cos(2*t*(w+wp))-9*np.sqrt(5)*A34*A38*np.pi*w**2*np.cos(2*t*(w+wp))+9*np.sqrt(5)*A32*A39*np.pi*w**2*np.cos(2*t*(w+wp))
    +64*A31**2*w*wp*np.cos(2*t*(w+wp))-72*A310**2*w*wp*np.cos(2*t*(w+wp))+64*A32**2*w*wp*np.cos(2*t*(w+wp))-64*A33**2*w*wp*np.cos(2*t*(w+wp))-64*A34**2*w*wp*np.cos(2*t*(w+wp))
    +384*A35**2*w*wp*np.cos(2*t*(w+wp))-384*A36**2*w*wp*np.cos(2*t*(w+wp))+72*A37**2*w*wp*np.cos(2*t*(w+wp))-72*A38**2*w*wp*np.cos(2*t*(w+wp))
    +72*A39**2*w*wp*np.cos(2*t*(w+wp))-18*np.sqrt(5)*A310*A33*np.pi*w*wp*np.cos(2*t*(w+wp))
    +18*np.sqrt(5)*A31*A37*np.pi*w*wp*np.cos(2*t*(w+wp))-18*np.sqrt(5)*A34*A38*np.pi*w*wp*np.cos(2*t*(w+wp))
    +18*np.sqrt(5)*A32*A39*np.pi*w*wp*np.cos(2*t*(w+wp))
    +32*A31**2*wp**2*np.cos(2*t*(w+wp))-36*A310**2*wp**2*np.cos(2*t*(w+wp))
    +32*A32**2*wp**2*np.cos(2*t*(w+wp))-32*A33**2*wp**2*np.cos(2*t*(w+wp))-32*A34**2*wp**2*np.cos(2*t*(w+wp))
    +192*A35**2*wp**2*np.cos(2*t*(w+wp))-192*A36**2*wp**2*np.cos(2*t*(w+wp))+36*A37**2*wp**2*np.cos(2*t*(w+wp))-36*A38**2*wp**2*np.cos(2*t*(w+wp))
    +36*A39**2*wp**2*np.cos(2*t*(w+wp))-9*np.sqrt(5)*A310*A33*np.pi*wp**2*np.cos(2*t*(w+wp))
    +9*np.sqrt(5)*A31*A37*np.pi*wp**2*np.cos(2*t*(w+wp))-9*np.sqrt(5)*A34*A38*np.pi*wp**2*np.cos(2*t*(w+wp))+9*np.sqrt(5)*A32*A39*np.pi*wp**2*np.cos(2*t*(w+wp))
    +64*A11*A31*w**2*np.cos(t*(2*w+wp))+64*A12*A32*w**2*np.cos(t*(2*w+wp))-64*A13*A33*w**2*np.cos(t*(2*w+wp))-64*A14*A34*w**2*np.cos(t*(2*w+wp))
    +384*A15*A35*w**2*np.cos(t*(2*w+wp))-384*A16*A36*w**2*np.cos(t*(2*w+wp))-9*np.sqrt(5)*A13*A310*np.pi*w**2*np.cos(t*(2*w+wp))
    +9*np.sqrt(5)*A11*A37*np.pi*w**2*np.cos(t*(2*w+wp))-9*np.sqrt(5)*A14*A38*np.pi*w**2*np.cos(t*(2*w+wp))+9*np.sqrt(5)*A12*A39*np.pi*w**2*np.cos(t*(2*w+wp))
    +64*A11*A31*w*wp*np.cos(t*(2*w+wp))+64*A12*A32*w*wp*np.cos(t*(2*w+wp))-64*A13*A33*w*wp*np.cos(t*(2*w+wp))-64*A14*A34*w*wp*np.cos(t*(2*w+wp))
    +384*A15*A35*w*wp*np.cos(t*(2*w+wp))-384*A16*A36*w*wp*np.cos(t*(2*w+wp))-9*np.sqrt(5)*A13*A310*np.pi*w*wp*np.cos(t*(2*w+wp))
    +9*np.sqrt(5)*A11*A37*np.pi*w*wp*np.cos(t*(2*w+wp))-9*np.sqrt(5)*A14*A38*np.pi*w*wp*np.cos(t*(2*w+wp))+9*np.sqrt(5)*A12*A39*np.pi*w*wp*np.cos(t*(2*w+wp))
    +64*A11*A13*w**2*np.sin(2*t*w)+64*A12*A14*w**2*np.sin(2*t*w)+384*A15*A16*w**2*np.sin(2*t*w)+192*A110*A18*w**2*np.sin(2*t*w)
    +192*A17*A19*w**2*np.sin(2*t*w)-64*A13*A31*w**2*np.sin(t*wp)-64*A14*A32*w**2*np.sin(t*wp)+64*A11*A33*w**2*np.sin(t*wp)
    +64*A12*A34*w**2*np.sin(t*wp)-384*A16*A35*w**2*np.sin(t*wp)
    +384*A15*A36*w**2*np.sin(t*wp)+9*np.sqrt(5)*A11*A310*np.pi*w**2*np.sin(t*wp)-9*np.sqrt(5)*A13*A37*np.pi*w**2*np.sin(t*wp)
    +9*np.sqrt(5)*A12*A38*np.pi*w**2*np.sin(t*wp)-9*np.sqrt(5)*A14*A39*np.pi*w**2*np.sin(t*wp)-64*A13*A31*w*wp*np.sin(t*wp)-64*A14*A32*w*wp*np.sin(t*wp)
    +64*A11*A33*w*wp*np.sin(t*wp)
    +64*A12*A34*w*wp*np.sin(t*wp)-384*A16*A35*w*wp*np.sin(t*wp)+384*A15*A36*w*wp*np.sin(t*wp)
    +9*np.sqrt(5)*A11*A310*np.pi*w*wp*np.sin(t*wp)-9*np.sqrt(5)*A13*A37*np.pi*w*wp*np.sin(t*wp)
    +9*np.sqrt(5)*A12*A38*np.pi*w*wp*np.sin(t*wp)-9*np.sqrt(5)*A14*A39*np.pi*w*wp*np.sin(t*wp)+128*A21*A22*wp**2*np.sin(2*t*wp)+192*A23*A25*wp**2*np.sin(2*t*wp)
    +192*A24*A26*wp**2*np.sin(2*t*wp)+64*A31*A33*w**2*np.sin(2*t*(w+wp))+64*A32*A34*w**2*np.sin(2*t*(w+wp))+384*A35*A36*w**2*np.sin(2*t*(w+wp))+72*A310*A37*w**2*np.sin(2*t*(w+wp))
    +72*A38*A39*w**2*np.sin(2*t*(w+wp))+9*np.sqrt(5)*A31*A310*np.pi*w**2*np.sin(2*t*(w+wp))+9*np.sqrt(5)*A33*A37*np.pi*w**2*np.sin(2*t*(w+wp))
    +9*np.sqrt(5)*A32*A38*np.pi*w**2*np.sin(2*t*(w+wp))+9*np.sqrt(5)*A34*A39*np.pi*w**2*np.sin(2*t*(w+wp))+128*A31*A33*w*wp*np.sin(2*t*(w+wp))+128*A32*A34*w*wp*np.sin(2*t*(w+wp))
    +768*A35*A36*w*wp*np.sin(2*t*(w+wp))+144*A310*A37*w*wp*np.sin(2*t*(w+wp))+144*A38*A39*w*wp*np.sin(2*t*(w+wp))+18*np.sqrt(5)*A31*A310*np.pi*w*wp*np.sin(2*t*(w+wp))
    +18*np.sqrt(5)*A33*A37*np.pi*w*wp*np.sin(2*t*(w+wp))+18*np.sqrt(5)*A32*A38*np.pi*w*wp*np.sin(2*t*(w+wp))+18*np.sqrt(5)*A34*A39*np.pi*w*wp*np.sin(2*t*(w+wp))
    +64*A31*A33*wp**2*np.sin(2*t*(w+wp))+64*A32*A34*wp**2*np.sin(2*t*(w+wp))+384*A35*A36*wp**2*np.sin(2*t*(w+wp))+72*A310*A37*wp**2*np.sin(2*t*(w+wp))
    +72*A38*A39*wp**2*np.sin(2*t*(w+wp))+9*np.sqrt(5)*A31*A310*np.pi*wp**2*np.sin(2*t*(w+wp))+9*np.sqrt(5)*A33*A37*np.pi*wp**2*np.sin(2*t*(w+wp))
    +9*np.sqrt(5)*A32*A38*np.pi*wp**2*np.sin(2*t*(w+wp))+9*np.sqrt(5)*A34*A39*np.pi*wp**2*np.sin(2*t*(w+wp))+64*A13*A31*w**2*np.sin(t*(2*w+wp))+64*A14*A32*w**2*np.sin(t*(2*w+wp))
    +64*A11*A33*w**2*np.sin(t*(2*w+wp))+64*A12*A34*w**2*np.sin(t*(2*w+wp))+384*A16*A35*w**2*np.sin(t*(2*w+wp))+384*A15*A36*w**2*np.sin(t*(2*w+wp))
    +9*np.sqrt(5)*A11*A310*np.pi*w**2*np.sin(t*(2*w+wp))+9*np.sqrt(5)*A13*A37*np.pi*w**2*np.sin(t*(2*w+wp))+9*np.sqrt(5)*A12*A38*np.pi*w**2*np.sin(t*(2*w+wp))
    +9*np.sqrt(5)*A14*A39*np.pi*w**2*np.sin(t*(2*w+wp))+64*A13*A31*w*wp*np.sin(t*(2*w+wp))+64*A14*A32*w*wp*np.sin(t*(2*w+wp))+64*A11*A33*w*wp*np.sin(t*(2*w+wp))
    +64*A12*A34*w*wp*np.sin(t*(2*w+wp))+384*A16*A35*w*wp*np.sin(t*(2*w+wp))+384*A15*A36*w*wp*np.sin(t*(2*w+wp))+9*np.sqrt(5)*A11*A310*np.pi*w*wp*np.sin(t*(2*w+wp))
    +9*np.sqrt(5)*A13*A37*np.pi*w*wp*np.sin(t*(2*w+wp))+9*np.sqrt(5)*A12*A38*np.pi*w*wp*np.sin(t*(2*w+wp))+9*np.sqrt(5)*A14*A39*np.pi*w*wp*np.sin(t*(2*w+wp)))


    
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

    idxsort = np.argsort(rs)
    rs=rs[idxsort]
    ts=ts[idxsort]
    ps=ps[idxsort]

    # interpolate these profiles using cubicsplines.
    tprofile = CubicSpline(rs, ts)
    pprofile = CubicSpline(rs, ps)

    return rtot, tprofile, pprofile
    
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
    pregen_r = None,
):
    """
    ts are times to evaluate power, in seconds.
    ecc is planet eccentricity
    r0 is an arbitrary small reference radius (m)
    p is planet orbital period, in days
    alpha is the orbital inclination, in radians
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
            rp, tprofile, pprofile = generate_profile(
                mpl=mp, teq=teq, cmf=cmf, log_feh=log_feh, co=co, tint=tint, fixed_radius=fixed_radius,
            )
    	else:
            rp, tprofile, pprofile = pregen_r, pregen_tprofile, pregen_pprofile


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
    y3 = find_ys(w + wp, rp, eta_use, r0=r0, eta_mult=eta_mult, temp_mult=temp_mult)
    coeffs = find_coeffs(rp, y1, y2, y3, alpha, m, d, ecc)
    a_vals = find_as(r_profile, y1, y2, y3, coeffs) 


    planet = Planet(t0=0, m0=0, ecc=ecc, a=a, ms=mstar)


    pos, _ = planet.calc_pos(ts / 24 / 3600)  # planet function takes times in days.
    ds = np.sqrt(np.sum(pos**2, axis=0)) / 100  # the planet function
    
    pind_profile = np.zeros(len(r_profile))

    # Loop over and calculate the power at each time.
    for i in range(len(ts)):

        teq = tstar * np.sqrt(rstar * RSOLAR / ds[i]) * (1 / 4) ** (1 / 4)
        teqs[i] = teq

        p_r = pvol(
            ts[i], w, wp, r_profile, rp, a_vals, eta_use, eta_mult=eta_mult, temp_mult=temp_mult
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
            rp, tprofile, pprofile = generate_profile(
                mpl=mp, teq=teq, cmf=cmf, log_feh=log_feh, co=co, tint=tint, fixed_radius=fixed_radius,
            )
    	else:
            rp, tprofile, pprofile = pregen_r, pregen_tprofile, pregen_pprofile


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

        p_r = pvol( # Evaluate the pvol, but with w=wp.
            ts[i],wp, wp,r_profile, rp, a_vals, eta_use, eta_mult=eta_mult, temp_mult=temp_mult
        )

        # save the power profile for the first value in the ts array
        if i==0:
            pind_profile = p_r
        pheat = scipy.integrate.simpson(p_r * r_profile**2, r_profile)

        tsurf = (pheat / (4 * np.pi * SB * rp**2) + teq**4) ** (1 / 4)
        dts[i] = tsurf - teq
        pinds[i] = pheat

    return pinds, dts, teqs, r_profile, pind_profile, eta_use(r_profile, rp)


