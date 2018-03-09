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
        self._coeffs.resize(p, self._maxL*2+1)
        self._maxP = p
        
        # Have to be clever with l indices to get correct array shape
        if l > self._maxL:
            # adding 2*(l-maxL) columns in middle of array
            # first column to add becomes maxL+1
            # last column added becomes l*2+1 - (maxL+1) = l*2-maxL
            # first column to move to end is maxL+1
            # last column to move to end is 2*maxL+1
            fstColSrc = self._maxL+1
            lstColSrc = 2*self._maxL+1
            fstColDest = 2*l - self._maxL + 1
            lstColDest = 2*l
            newCoeffs = np.zeros((self._maxP, l*2+1), dtype=complex)
            newCoeffs[:, :fstColSrc] = self._coeffs[:, :fstColSrc]
            newCoeffs[:, fstColDest:] = self._coeffs[:, fstColSrc:lstColSrc]
            self._coeffs = newCoeffs
        if l < self._maxL:
            # adding 2*(l-maxL) columns in middle of array
            # first column to move is 2*maxL+1-l
            # last column to move is  2*maxL+1
            # first column to move moves to l+1
            # last column to move moves to 2*l+1
            
            fstColSrc = 2*self._maxL+1-l
            lstColSrc = 2*self._maxL+1
            fstColDest = l+1
            lstColDest = 2*l+1
            newCoeffs = np.zeros((self._maxP, l*2+1), dtype=complex)
            newCoeffs[:, :l+2] = self._coeffs[:, :l+2]
            newCoeffs[:, fstColDest:] = self._coeffs[:, fstColSrc:lstColSrc]
            self._coeffs = newCoeffs
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
