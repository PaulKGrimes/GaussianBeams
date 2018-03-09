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

