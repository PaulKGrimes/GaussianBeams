# GaussianBeams.py
#
# Paul Grimes - March 2018
#
# Defines Gauss-Laguerre and modesets for modelling beams.  
# Both regular paraxial G-L modesets and modifed G-L modesets 
# after Tuovinen and Friberg 1992 are available.
#
# Functions implementing the calculations are in GaussianLaguerreModes.py

from GaussianLaguerreModes import *

class GaussLaguerreModeSet:
    """A set of Gauss-Laguerre modes with complex coefficients"""
    def __init__(self, k=1., w0=1., maxP = 0, maxL = 0):
        # Create a complex array holding the mode coefficients of the G-L modes
        # Indexing runs from p=0 to p=maxP in the first dimension and
        # l=0 to maxL then -maxL to l=-1 in the second.
        self._coeffs = np.zeros((maxP+1, 2*maxL+1), dtype=complex)
        self._coeffs[0][0] = complex(1., 0)
        
        self._k = k    # The wavenumber of the G-L modes
        self._w0 = w0  # The beam waist radius of the G-L modes
        self._maxP = 0 # The highest index of the axial modes included in the modeset
                       # axial mode index p is in the range 0 < p < maxP
        self._maxL = 0 # The highest absolute index of the azimuthal modes included in the modeset
                       # azimuthal mode index l is in the range -maxL < l < maxL
    
    @property
    def k(self):
        """return wavenumber k"""
        return self._k
     
    @k.setter
    def k(self, newK):
        """Set the value of k"""
        self._k = newK
        
    @property
    def w0(self):
        """return the beam waist radius w0"""
        return self._w0
        
    @w0.setter
    def w0(self, newW):
        """Set the value of w0"""
        self._w0 = newW
        
    def resizeModeSet(self, p, l):
        """Resize the array of mode coefficients"""
        # Don't have to do anything clever with p indices
        self._coeffs.resize(p, self._maxL)
        self._maxP = p
        
        # Have to be clever with l indices to get correct array shape
        self._coeffs.resize(self._maxP, l)
        if l > self._maxL:
            self._coeffs.resize(self._maxP, l)
            self._coeffs[:, -self._maxL:-1] = self._coeffs[:, self._maxL+1:-l-self._maxL]
            self._coeffs[:, -l:-self._maxL] = complex(0.0)
        if l < self._maxL:
            self._coeffs[:, l+1:-(self._maxL-l)] = self._coeffs[:, -l:-1]
            self._coeffs.resize(self._maxP, l)
        self._maxL = l
        
    
    @property
    def maxP(self):
        """return the maximum absolute index for the axial mode index"""
        return self._maxP
        
    @maxL.setter
    def maxP(self, p):
        """Set a new value for maxP"""
        # resize the self._coeffs array
        self.resizeModeSet(p, self._maxL)
    
    @property
    def maxL(self):
        """return the maximum absolute index for the azimuthal mode index"""
        return self._maxL
        
    @maxL.setter
    def maxL(self, l):
        """Set a new value for maxL"""
        # resize the self._coeffs array
        self.resizeModeSet(self._p, l)
