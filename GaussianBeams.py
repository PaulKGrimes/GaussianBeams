# GaussianBeams.py
#
# Paul Grimes - March 2018
#
# Defines Gauss-Laguerre and modesets for modelling beams.  
# Both regular paraxial G-L modesets and modifed G-L modesets 
# after Tuovinen and Friberg 1992 are available.
#
# Functions implementing the calculations are in GaussianLaguerreModes.py

import GaussianLaguerreModes as Glm
import ModifiedGaussianLaguerreModes as modGlm

class GaussLaguerreModeBase(object):
    """The base class of the GLM and modified GLM classes.  This base class implements the common parameters and
    handles the storage and manipulation of the mode coefficients"""
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

        
class GaussLaguerreModeSet(GaussLaguerreModeBase):
    """A class holding a set of Gauss-Laguerre modes, defined in the paraxial limit."""
    def __init__(self, k=1., w0=1., maxP = 0, maxL = 0):
        super(GaussLaguerreModeSet, self).__init__(k, w0, maxP, maxL)
        
    def field(self, rho, phi, z, p=None, l=None):
        """Return the value of the field at rho, phi, z; either for the sum of all modes (p, l) = None,
        for a specified axial mode p (sum over azimuthal modes), or for a specific (p, l) mode."""
        if p!=None and l!=None:
            # We are after a specific mode
            return self_coeffs[p,l]* GLM.Glm(rho, phi, z, self.k, self.w0, p=p, l=l)
        elif p!=None and l==None:
            # We are after the sum of all azimuthal modes in an axial mode
            result = np.zeros_like(np.meshgrid(rho, phi, z), dtype=np.complex)
            for l in range(-self.maxL, self.maxL+1):
                result += self.field(rho, phi, z, p=p, l=l)
            return result
        elif p==None and l == None:
            # We are after the sum of all modes.
            result = np.zeros_like(np.meshgrid(rho, phi, z), dtype=np.complex)
            for p in range(0, self.maxP+1):
                result += self.field(rho, phi, z, p=p, l=None)
            return result
        else:
            # Shouldn't get here
            raise ValueError, "GaussLaguerreModeSet.field: must set mode index p if mode index l is set"
        
    def farField(self, theta, phi, p=None, l=None):
        """Return the value of the far field at theta, phi; either for the sum of all modes (p, l) = None,
        for a specified axial mode p (sum over azimuthal modes), or for a specific (p, l) mode."""
                if p!=None and l!=None:
            # We are after a specific mode
            return self_coeffs[p,l]* GLM.Glm_ff(theta, phi, self.k, self.w0, p=p, l=l)
        elif p!=None and l==None:
            # We are after the sum of all azimuthal modes in an axial mode
            result = np.zeros_like(np.meshgrid(theta, phi), dtype=np.complex)
            for l in range(-self.maxL, self.maxL+1):
                result += self.field(theta, phi, p=p, l=l)
            return result
        elif p==None and l == None:
            # We are after the sum of all modes.
            result = np.zeros_like(np.meshgrid(theta, phi), dtype=np.complex)
            for p in range(0, self.maxP+1):
                result += self.field(theta, phi, p=p, l=None)
            return result
        else:
            # Shouldn't get here
            raise ValueError, "GaussLaguerreModeSet.farField: must set mode index p if mode index l is set"
        


class ModifiedGaussLaguerreModeSet(GaussLaguerreModeBase):
    """A class holding a set of modified Gauss-Laguerre modes, using the definition Tuovinen (1992)"""
    def __init__(self, k=1., w0=1., maxP = 0, maxL = 0):
        super(GaussLaguerreModeSet, self).__init__(k, w0, maxP, maxL)
        
    def field(self, rho, phi, z, p=None, l=None):
        """Return the value of the field at rho, phi, z; either for the sum of all modes (p, l) = None,
        for a specified axial mode p (sum over azimuthal modes), or for a specific (p, l) mode."""
        if p!=None and l!=None:
            # We are after a specific mode
            return self_coeffs[p,l]* modGLM.Glm(rho, phi, z, self.k, self.w0, p=p, l=l)
        elif p!=None and l==None:
            # We are after the sum of all azimuthal modes in an axial mode
            result = np.zeros_like(np.meshgrid(rho, phi, z), dtype=np.complex)
            for l in range(-self.maxL, self.maxL+1):
                result += self.field(rho, phi, z, p=p, l=l)
            return result
        elif p==None and l == None:
            # We are after the sum of all modes.
            result = np.zeros_like(np.meshgrid(rho, phi, z), dtype=np.complex)
            for p in range(0, self.maxP+1):
                result += self.field(rho, phi, z, p=p, l=None)
            return result
        else:
            # Shouldn't get here
            raise ValueError, "ModifiedGaussLaguerreModeSet.field: must set mode index p if mode index l is set"
        
    def farField(self, theta, phi, p=None, l=None):
        """Return the value of the far field at theta, phi; either for the sum of all modes (p, l) = None,
        for a specified axial mode p (sum over azimuthal modes), or for a specific (p, l) mode."""
                if p!=None and l!=None:
            # We are after a specific mode
            return self_coeffs[p,l]* modGLM.Glm_ff(theta, phi, self.k, self.w0, p=p, l=l)
        elif p!=None and l==None:
            # We are after the sum of all azimuthal modes in an axial mode
            result = np.zeros_like(np.meshgrid(theta, phi), dtype=np.complex)
            for l in range(-self.maxL, self.maxL+1):
                result += self.field(theta, phi, p=p, l=l)
            return result
        elif p==None and l == None:
            # We are after the sum of all modes.
            result = np.zeros_like(np.meshgrid(theta, phi), dtype=np.complex)
            for p in range(0, self.maxP+1):
                result += self.field(theta, phi, p=p, l=None)
            return result
        else:
            # Shouldn't get here
            raise ValueError, "ModifiedGaussLaguerreModeSet.farField: must set mode index p if mode index l is set"
        
