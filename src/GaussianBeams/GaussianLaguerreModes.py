# GaussianLaguerreModes.py
#
# Paul Grimes - March 2018
#
# Defines Gauss-Laguerre modes and functions for decomposing fields
# into Gauss-Laguerre mode sets
#
# Both traditional paraxial G-L modes and the modified G-L modes due to Tuovinen and Friberg are available
#
import numpy as np

from scipy.constants import c
from scipy.special import jn, jn_zeros
from math import factorial

j = 1j

def k(self, f, e_r=1.0):
    """return the wavenumber for a wave at frequency f in medium with dielectric constant e_r (default e_r = 1.0)"""
    return 2*np.pi*f/c

# Everything below here copied from ModifiedGaussianLaguerreModes and needs altering
def alpha(rho, ww, FF):
    """return the alpha parameter for the modified Gauss-Laguerre modes"""
    return np.sqrt(2)*rho/(ww*FF)

def alpha_ff(theta, w0, kk):
    """return the farfield alpha parameter for the modified Gauss-Laguerre modes"""
    return 1./np.sqrt(2) * kk * w0 * np.sin(theta)

def F(rho, z, bb):
    r = R(z, bb)
    return np.sqrt(1+(rho/r)**2)

def b(w0, kk):
    """return the value of b"""
    return kk*(w0**2)/2.0

def Phi0(z, bb):
    return np.arctan(z/bb)

def R(z, bb):
    return z*(1+(bb/z)**2)

def w(z, w0, bb):
    return w0*np.sqrt(1+(z/bb)**2)

def F(rho, z, bb):
    r = R(z, bb)
    return np.sqrt(1+(rho/r)**2)

def Lpl(x, p=0, l=0):
    """return the value of the Gaussian-Laguerre polynomial at x"""
    return sp.special.eval_genlaguerre(p, abs(l), x)

def Cpl(p=0, l=0):
    """return the normalization factor for the p,lth G-L mode"""
    return np.sqrt(4*np.pi)*np.sqrt((2*factorial(p)/(np.pi*factorial(abs(l)+p))))

def sign(theta):
    """return the sign to use in the far field exponent.
    +1 for theta between 0 and 90 deg
    -1 for theta between 90 and 180 deg"""
    return np.sign(np.pi/2 - theta)

def exponent(rho, phi, z, w0, kk, p=0, l=0):
    """return the exponent of e in Glm mode"""
    bb = b(w0, kk)
    ww = w(z, w0, bb)
    FF = F(rho, z, bb)
    aa = alpha(rho, ww, FF)
    RR = R(z, bb)
    PP = Phi0(z, bb)

    return -(aa**2)/2 -j*kk*RR*(FF-1) -j*kk*z +j*(2*p + abs(l) + 1)*PP +j*l*phi

def exponent_ff(theta, phi, a, p=0, l=0):
    """return the farfield exponent of e in Glm_ff mode"""
    theta_d = theta+np.pi*1e-16
    exp = -(a**2)/2 + sign(theta_d)*j*(2*p + abs(l) + 1)*np.pi/2 +j*l*phi
    return exp

def Glm(rho, phi, z, kk, w0, p=0, l=0):
    """return the value of the modified G-L mode"""
    bb = b(w0, kk)
    ww = w(z, w0, bb)
    FF = F(rho, z, bb)
    aa = alpha(rho, ww, FF)
    cosTh = 1.0/FF

    return Cpl(p, l) * (1+cosTh)/2 * 1/(kk*ww*FF) * aa**abs(l) * Lpl(aa**2, p, l) * np.exp(exponent(rho, phi, z, kk, w0, p, l))

def Glm_ff(theta, phi, kk, w0, p=0, l=0):
    """return the Farfield value of the modified G-L mode"""
    aa = alpha_ff(theta, kk, w0)
    return Cpl(p, l) * (1+np.abs(np.cos(theta)))/4 * kk*w0 * aa**abs(l) * Lpl(aa**2, p, l) * np.exp(exponent_ff(theta, phi, aa, p, l))
