import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from variables import mu,E_k,L_k,Q_k,a


def s_potential(r, l,m=1):
    """
    A function that returns the equation for the Schwarzschild Effective Potential;
    plotting it allows us to see the event horizon of a non-rotating black hole.
    
    Parameters:
    ------------------------------------------
    l : specific conserved angular momentum
    r : radius of black hole
    m : estimated mass of black hole, usually set aa 1 if G,C = 1
    """

    return (1 - ((2 * m) / r)) * (l**2 / (2 * (r**2))) - (m/r)


def effectivep_fourtwo(epsilon):
    """
    Calculates the point at where the potential dips, 
    showing us the start and end of the black hole. 
    """
    return (epsilon**2 - 1) / 2


def effective_p(r):
    """
    Schwarzschild effective potential function of l = 4.2,
    added with the limit of the bound orbits as calculated at E = 0.972.
    This will be the function that we will find the roots of, to obtain
    """
    l = 4.2
    return s_potential(r, l) - -0.026635500000000034


def kerr_limits(r):
    """
    A function that returns the equation for the effective potential 
    of a spinning black hole. 
    Parameters
    ---------------------
    E_k : Specific energy
    L_k : Specific angular momentum
    Q_k : Carter's constant
    a : Spin
    """
    return ((E_k**2 - mu**2) * r**4 + 2 * mu**2 * r**3 + 
            (a**2 * (E_k**2 - mu**2) - Q_k - L_k**2)* r**2 + 
            2 * (Q_k + (L_k - a * E_k)**2)* r - a**2 * Q_k)


def sph2car(r, theta, p):
    """
    A function converting Spherical coordinates to Cartesian coordinates 
    for simpler plotting in 2D and 3D. 

    x, y, z : Cartesian Coordinates
    r       : spherical coordinate r
    theta   : spherical coordinate theta
    p       : spherical coordinate p
    """
    x = r * np.sin(theta) * np.cos(p)
    y = r * np.sin(theta) * np.sin(p)
    z = r * np.cos(theta)

    return list(x), list(y), list(z)


def sch_geod(t, vector,m=1):
    """
    The Schwarzschild Geodesic in a function; 
    this will be the equations we are integrating for orbits 
    around a non-rotating black hole.

    Parameters:
    -------------------------------------
    t  : initial point; array
    v  : tangent vector; array

    """
    values = np.zeros(shape=vector.shape, dtype=vector.dtype)
    r, theta = vector[1], vector[2]

    #dot = overdot
    tdot = vector[4]             
    rdot = vector[5]
    thetadot = vector[6]
    pdot = vector[7]

    #for simplicity
    r2 = r**2
    r4 = r**4

    values[:4] = vector[4:]

    values[4] = (2 * rdot * tdot * m) / (2 * m * r - r2)

    values[5] = - ( 4 * np.square(thetadot) * m * r**3 - 4 * np.square(thetadot) * m * r4 + np.square(thetadot) * r**5 - 4 * np.square(tdot) * m**3 + 4 * np.square(tdot) * m**2 * r + (np.square(rdot) - np.square(tdot)) * m * r**2 + (4 * np.square(pdot) * m**2 * r**3 - 4 * np.square(pdot) * m * r4 + np.square(pdot) * r**5) * np.sin(theta)**2) / (2 * m * r**3 - r4)

    values[6] = (np.square(pdot) * r * np.cos(theta) * np.sin(theta) - 2 * rdot * thetadot) / r

    values[7] = -2 * (pdot * thetadot * r * np.cos(theta) + pdot * rdot * np.sin(theta)) / (r * np.sin(theta))
    return values


def kerr_geod(t, vector):
    """
    The Kerr geodesic as a function and will be the main geodesic equation we are integrating to plot orbits around a rotating black hole with the spin (a) = 0.998.

       Parameters:
       -------------------------------------
       t  : initial point; array
       v  : tangent vector; array
    """
    values = np.zeros(shape=vector.shape, dtype=vector.dtype)
    r, theta = vector[1], vector[2]
    t_dot = vector[4]
    r_dot = vector[5]
    theta_dot = vector[6]
    p_dot = vector[7]
    a = 0.998

    #for simplicity
    r2, a2 = r**2, a**2
    r4, a4 = r**4, a**4
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_t2, cos_t2 = sin_t**2, cos_t**2
    cos_t4 = cos_t2**2
    sin2t, cos2t, cos4t = np.sin(2 * theta), np.cos(2 * theta), np.cos(4 * theta)

    sigma = r2 + a2 * cos_t2
    delta = r2 + a2 - 2 * r

    values[:4] = vector[4:]

    values[4] = (-2 * a2 * r * theta_dot * (-2 * (a4 + 2 * (-8 + r) * r2 * r + a2 * r * (-14 + 3 * r) + 
        a2 * (a2 + (-2 + r) * r) * cos2t) * sin2t * t_dot + 
        8 * a * (a2 + (-2 + r) * r) * cos_t * (a2 + 2 * r2 + 
        a2 * cos2t) * sin_t2 * sin_t * p_dot) + r_dot * ((3 * a4 * a2 - 6 * a4 * r + 3 * a4 * r2 + 24 * a2 * r2 * r - 8 * a2 * r4 - 
        8 * r4 * r2 + 4 * a2 * (a4 + a2 * r2 - 6 * r2 * r) * cos2t + 
        a4 * (a2 + r * (6 + r)) * cos4t) * t_dot - 
        16 * a * ((-r4 * (a2 + 3 * r2) + 
        a4 * (a2 - r2) * cos_t4) * sin_t2 - 
        a2 * r4 * (np.square(sin2t))) * p_dot))/(4 * (r2 + 
        a2 * cos_t2) * (2 * a2 * r2 * (2 + a2 - 2 * r + 
        r2) * cos_t2 + 
        a4 * (a2 + (-2 + r) * r) * cos_t4 + 
        r2 * ((-2 + r) * r2 * r + a2 * (4 + r2) - 8 * a2 * cos2t)))

    values[5] = (1/(np.power((sigma), 3))) *(
        (((np.square((sigma))) * (r *(-a2 + r) + a2 * (-1 + r) * cos_t2) * (np.square(r_dot))) / (a2 + (-2 + r) * r)) + 
        ((a2 + (-2 + r) * r) * (-r2 + a2 * cos_t2) * (np.square(t_dot))
        ) + (2 * a2 * cos_t * (np.square((sigma))) * sin_t * r_dot * theta_dot
        ) + (r * (a2 + (-2 + r) * r) * (np.square((r2 + a2 * cos_t2))) * (np.square(theta_dot))
        ) - (4 * a * (a2 + (-2 + r) * r) * (-r2 + a2 * cos_t2) * sin_t2 * t_dot * p_dot
        ) + ((a2 + (-2 + r) * r) * (-a2 * r2 + r4 * r + a2 * (a2 + 
        r2 + 2 * r2 * r) * cos_t2 + a4 * (-1 + r) * cos_t4) * sin_t2 * (np.square(p_dot))
        )
    )


    values[6] = (1/(np.power((sigma), 3))) * (-((
        a2 * cos_t * (np.square((sigma))) * sin_t * (np.square(r_dot)))/(
        a2 + (-2 + r) * r)) + a2 * r * sin2t * (np.square(t_dot)) - 
        2 * r * (np.square((sigma))) * r_dot * theta_dot + 
        a2 * cos_t * (np.square((sigma))) * sin_t * (np.square(theta_dot)) - 
        4 * a * r * (a2 + r2) * sin2t * t_dot * p_dot - 
        cos_t * sin_t * (-(np.square((sigma))) * (a2 - 
        2 * r + r2 + (2 * r * (a2 + r2))/(sigma)) - 
        2 * a2 * r * (a2 + r2) * sin_t2) * (np.square(p_dot)))

    values[7] = -((2 * (theta_dot * (-2 * a * r * (a2 + (-2 + r) * r) * (a2 + 2 * r2 + 
        a2 * cos2t) * (cos_t / sin_t) * t_dot + ((a2 * r2 * (a2 * (-4 + 3 * r2) + 
        r2 * (4 - 6 * r + 3 * r2)) * cos_t2 + 
        a4 * r2 * (4 + 3 * a2 - 6 * r + 3 * r2) * cos_t4 + 
        a4 * a2 * (a2 + (-2 + r) * r) * cos_t4 * cos_t2 + 
        r2 * ((-2 + r) * r4 * r + a4 * (6 + r) + 
        a2 * r2 * (2 + r + r2) - 
        a2 * (6 + r) * (a2 + r2) * cos2t)) * (cos_t / sin_t) + 
        2 * a4 * r * (a2 + 
        r2) * cos_t2 * cos_t * sin_t) * p_dot) + r_dot * ((2 * a * r4 - 2 * a4 * a * cos_t4) * t_dot + (2 * a2 * r2 * r - a2 * r4 - 2 * r4 * r2 + r4 * r2 * r - 
        a2 * r * (2 * a2 + r2 * (2 + 3 * r - 3 * r2)) * cos_t2 + 
        a4 * (a2 + r * (2 - 2 * r + 3 * r2)) * cos_t4 + 
        a4 * a2 * (-1 + r) * cos_t4 * cos_t2 - 
        8 * a2 * r2 * r * sin_t2 + 
        2 * a4 * r * (np.square(sin2t))) * p_dot)))/((r2 + 
        a2 * cos_t2) * (2 * a2 * r2 * (2 + a2 - 2 * r + 
        r2) * cos_t2 + 
        a4 * (a2 + (-2 + r) * r) * cos_t4 + 
        r2 * ((-2 + r) * r2 * r + a2 * (4 + r2) - 8 * a2 * cos2t))))
    
    return values


def newton_step(f, fp, x0):
    """Perform one step of the Newton-Raphson algorithm."""

    x1 = x0 - ( f(x0) / fp(x0) )
    return (x1)

def CD(f, x, h=1e-5):
    """Estimate the derivative f'(x) using the central difference algorithm with step size h."""
    
    return (f(x + h/2) -  f(x - h/2))/h

def newton_CD_step(f, x0):
    """Combined function of the Newton Step method and Central difference method"""
    
    return x0 - f(x0) / CD(f, x0)


def root_finder(f,epsilon,x,Nitermax,dif):
    n=0
    while dif > epsilon and n <= Nitermax:
        xn1 = x
        x = newton_CD_step(f, x)
        dif = abs(x - xn1)
        n += 1
    return x,n

