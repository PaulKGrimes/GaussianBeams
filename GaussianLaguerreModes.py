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
import scipy as sp

from scipy.constants import c
from scipy.special import jn, jn_zeros
from scipy.misc import factorial
from scipy.integrate import simps


j = 1j

def Epl(rho, phi, k, w, R, p=0, l=0):
    """Calculate complex value of normalized Gauss-Laguerre mode at rho, phi, z in
    cylindrical coordinates

    Arguments:
        rho: radial distance - numpy array or float.
        phi: azimuthal angle - numpy array or float.
        k:   wavenumber of beam - float.
        w:   beam radius - float.
        R:   radius of phase curvature - float.
        p:   axial mode index - integer.
        l:   azimuthal mode index - integer
    Returns:
        complex value of field - numpy array of complex values"""
    # Convert rho and phi into meshgrids
    rho, phi = make_grids(rho, phi)

    lm = 2*np.pi/k
    z = z_from_wR(w, R, lm)
    w0 = w0_from_wR(w, R, lm)
    phi0 = phi0_from_w0z(w0, z, lm)

    Eamp = Epl_amp(rho, phi, k, w, R, p, l)
    Ephase = Epl_phase(rho, phi, k, w, R, p, l)
    Ephi = Epl_phi(rho, phi, l)
    return Eamp*Ephase*Ephi


def Epl_amp(rho, phi, k, w, R, p=0, l=0):
    """Calculate the amplitude of normalized Gauss-Laguerre mode at rho, phi in
    cylindrical coordinates

    Arguments:
        rho: radial distance - numpy array or float.
        phi: azimuthal angle - numpy array or float.
        k:   wavenumber of beam - float.
        w:   beam radius - float.
        R:   radius of phase curvature - float.
        p:   axial mode index - integer.
        l:   azimuthal mode index - integer
    Returns:
        amplitude of the field - numpy array of float values"""
    # Convert rho and phi into meshgrids
    rho, phi = make_grids(rho, phi)

    lm = 2*np.pi/k
    z = z_from_wR(w, R, lm)
    w0 = w0_from_wR(w, R, lm)

    amp_exponent = -rho**2/w**2

    Spl = Sr(rho, w, p, l)

    nrm = norm(w, p, l)

    return nrm*Spl*np.exp(amp_exponent)

def Epl_phase(rho, phi, k, w, R, p=0, l=0):
    """Calculate the radial dependent phase of normalized Gauss-Laguerre mode at rho, phi in
    cylindrical coordinates

    Arguments:
        rho: radial distance - numpy array or float.
        phi: azimuthal angle - numpy array or float.
        k:   wavenumber of beam - float.
        w:   beam radius - float.
        R:   radius of phase curvature - float.
        p:   axial mode index - integer.
        l:   azimuthal mode index - integer
    Returns:
        phase factor of the field - numpy array of float values"""
    rho, phi = make_grids(rho, phi)

    lm = 2*np.pi/k
    z = z_from_wR(w, R, lm)
    w0 = w0_from_wR(w, R, lm)
    phi0 = phi0_from_w0z(w0, z, lm)

    phase_exponent = - j*k*z - j*k*rho**2/(2*R) - j*(2*p+np.abs(l)+1)*phi0

    return np.exp(phase_exponent)

def Epl_phi(rho, phi, l=0):
    """Calculate the azimuthal dependent phase of normalized Gauss-Laguerre mode at rho, phi in
    cylindrical coordinates

    Arguments:
        rho: radial distance - numpy array or float.
        phi: azimuthal angle - numpy array or float.
        l:   azimuthal mode index - integer
    Returns:
        phase factor of the field - numpy array of float values"""
    rho, phi = make_grids(rho, phi)

    phi_exponent = -j*l*phi

    return np.exp(phi_exponent)

def make_grids(rho, phi):
    """Convert rho and phi into meshgrids."""
    if type(rho) is float or type(rho) is int:
        rho = np.array([rho])
    if type(phi) is float or type(phi) is int:
        phi = np.array([phi])

    if len(rho.shape) == 1 and len(phi.shape) == 1:
        rho, phi = np.meshgrid(rho, phi)

    return rho, phi

def norm(w, p=0, l=0):
    """ Calculate the normalization constant for the p, l G-L mode with beam width w.

    Arguments:
        w:  beam radius - float.
        p:  axial mode index - integer.
        l:  azimuthal mode index - integer.

    Returns:
        normalization constant - float.
        """
    return np.sqrt(2/(np.pi) * factorial(p)/factorial(p + np.abs(l))) / w


def Lpl(x, p=0, l=0):
    """return the value of the Gaussian-Laguerre polynomial at x

    Arguments:
        x: value to evaluate function at - float or numpy array
        p:   axial mode index
        l:   azimuthal mode index

    Returns:
        float or numpy array (according to type of rho) containing the function value."""
    return sp.special.eval_genlaguerre(p, np.abs(l), x)


def Sr(rho, w, p=0, l=0):
    """Calculate the radial function of the p, l, G-L mode.

    Arguments:
        rho: radial distance - numpy array or float
        w:   beam radius - float
        p:   axial mode index
        l:   azimuthal mode index

    Returns:
        float or numpy array (according to type of rho) containing the radial
        distribution."""
    return (np.sqrt(2)*rho/w)**np.abs(l) * Lpl(2*rho**2 / w**2, p, l)


def phi0_from_w0z(w0, z ,lm):
    """Calculate the beam phase slippage from beam waist radius, axial distance,
    and wavelength.

    Arguments:
        w0: beam waist radius - float.
        z: axial distance along beam - float.
        lm: wavelength - float.
        All arguments assumed to be in the same units.

    Returns:
        phi0: beam phase slippage in radians - float.
        """
    phi0 = np.arctan((lm*z)/(np.pi*w0**2))
    return phi0

def Aint(data, rho, phi, k, w, R, p=0, l=0):
    """Return the integrand of overlap integral between the generalized
    Gaussian-Laguerre mode with beam radius w, radius of phase curvature R,
    given data over radial distance rho and angles phi.

    Arguments:
        data: numpy array containing complex field data.
        rho: numpy array containing rho values of data.
        phi: numpy array containing phi values of data.
        k: wavenumber of beam/data - float.
        w: beam radius of Gaussian mode - float.
        R: radius of phase curvature of Gaussian mode - float.
        p: axial mode index - integer.
        l: azimuthal mode index - integer.
    Returns:
        numpy array containing complex values of integrand."""
    return data*np.conj(Epl(rho, phi, k, w, R, p, l))*np.abs(rho)

def Apl(data, rho, phi, k, w, R, p=0, l=0):
    """Return the overlap integral between the generalized
    Gaussian-Laguerre mode with beam radius w, radius of phase curvature R,
    given data over radial distance rho and angles phi.

    Arguments:
        data: numpy array containing complex field data.
        rho: numpy array containing rho values of data.
        phi: numpy array containing phi values of data.
        k: wavenumber of beam/data - float.
        w: beam radius of Gaussian mode - float.
        R: radius of phase curvature of Gaussian mode - float.
        p: axial mode index - integer.
        l: azimuthal mode index - integer.
    Returns:
        complex value of integral."""
    # Integrate over phi first, using the averaging method to deal with end points
    # as most horn patterns will have rotational symmetry.
    # Then integrate over rho, using the "first" method to deal with endpoints, as
    # most horn patterns will go to zero at high rho, and we will probably have
    # rho having increasing radius
    return simps(simps(Aint(data, rho, phi, k, w, R, p, l), phi, axis=0, even="avg"), rho, even="first")

# The following functions allow conversions from pairs of Gaussian beam parameters
# (w0, w, R, z) to other Gaussian beam parameters

def w_from_w0z(w0, z, lm):
    """Calculate the beam radius at z from beam waist radius, z,
    and wavelength.

    Arguments:
        w0: beam waist radius - float.
        z: axial distance along beam - float.
        lm: wavelength - float.
        All arguments assumed to be in the same units.

    Returns:
        w: beam radius at z in same units as input - float.
    """
    w = w0*np.sqrt(1 + ((lm*z)/(np.pi*w0**2))**2)
    return w

def R_from_w0z(w0, z, lm):
    """Calculate the radius of phase curvature at z from beam waist radius, z,
    and wavelength.

    Arguments:
        w0: beam waist radius - float.
        z: axial distance along beam - float.
        lm: wavelength - float.
        All arguments assumed to be in the same units.

    Returns:
        R: radius of phase curvature at z in same units as input - float.
    """
    R = z*(1 + ((np.pi*w0**2)/(lm*z))**2)
    return R

def w0_from_zR(z, R, lm):
    """Calculate the beam waist radius from distance along beam, radius of phase curvature
    and wavelength.

    Arguments:
        z: axial distance along beam - float.
        R: radius of phase curvature - float.
        lm: wavelength - float.
        All arguments assumed to be in the same units.

    Returns:
        w0: beam waist radius in same units as input - float.
    """
    w0 = np.sqrt(lm/np.pi * np.sqrt(z*(R-z)))
    return w0

def w0_from_zw(z, w, lm):
    """Calculate the beam waist radius from distance along beam, beam radius
    and wavelength.

    Arguments:
        z: axial distance along beam - float.
        w: beam radius at z - float.
        lm: wavelength - float.
        All arguments assumed to be in the same units.

    Returns:
        w0: beam waist radius in same units as input - float.
    """
    w02 = w**2/2 * (1 + np.sqrt(1-(2*lm*z/(np.pi*w**2))**2))
    if w02 < 0:
        w02 = w**2/2 * (1 - np.sqrt(1-(2*lm*z/(np.pi*w**2))**2))

    return np.sqrt(w02)

def z_from_w0w(w0, w, lm):
    """Calculate the distance along beam from beam waist radius, beam radius
    and wavelength.

    Arguments:
        w0: beam waist radius - float.
        w: beam radius at z - float.
        lm: wavelength - float.
        All arguments assumed to be in the same units.

    Returns:
        z: axial distance along beam in same units as input - float.
    """
    z = np.pi*w0/lm * np.sqrt(w**2-w0**2)
    return z

def z_from_w0R(w0, R, lm):
    """Calculate the distance along beam from beam waist radius, radius of phase curvature
    and wavelength.

    Arguments:
        w0: beam waist radius - float.
        R: radius of phase curvature at z - float.
        lm: wavelength - float.
        All arguments assumed to be in the same units.

    Returns:
        z: axial distance along beam in same units as input - float.
    """
    z = R/2 * (1 - np.sqrt(1 - ((2*np.pi*w0**2)/(lm*R))**2))
    return z

def w0_from_wR(w, R, lm):
    """Calculate the beam waist radius from beam radius, radius of phase curvature
    and wavelength.

    Arguments:
        w: beam radius - float.
        R: radius of phase curvature - float.
        lm: wavelength - float.
        All arguments assumed to be in the same units.

    Returns:
        w0: beam waist radius in same units as input - float.
    """
    w0 = w / np.sqrt(1 + ((np.pi*w**2)/(lm*R))**2)
    return w0

def z_from_wR(w, R, lm):
    """Calculate the axial distance from beam radius, radius of phase curvature
    and wavelength.

    Arguments:
        w: beam radius - float.
        R: radius of phase curvature - float.
        lm: wavelength - float.
        All arguments assumed to be in the same units.

    Returns:
        z: axial distance along beam in units of inputs - float."""
    z = R / (1 + ((lm*R)/(np.pi*w**2))**2)
    return z
