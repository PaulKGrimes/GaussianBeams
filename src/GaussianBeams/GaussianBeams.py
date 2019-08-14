# GaussianBeams.py
#
# Paul Grimes - March 2018
#
# Defines Gauss-Laguerre and modesets for modelling beams.
# Both regular paraxial G-L modesets and modifed G-L modesets
# after Tuovinen and Friberg 1992 are available.
#
# Functions implementing the calculations are in GaussianLaguerreModes.py

import numpy as np
import scipy as sp

import GaussianLaguerreModes as Glm
import ModifiedGaussianLaguerreModes as modGlm

class GaussLaguerreModeBase(object):
    """The base class of the GLM and modified GLM classes.  This base class implements the common parameters and
    handles the storage and manipulation of the mode coefficients.

    Attributes:
        k: wavenumber of beam.
        w0: beam waist diameter of beam.
        maxP: number of axial modes in beam set.
        maxL: number of azimuthal modes in beam set."""
    def __init__(self, w0=1., k=1., maxP = 0, maxL = 0):
        # Create a complex array holding the mode coefficients of the G-L modes
        # Indexing runs from p=0 to p=maxP in the first dimension and
        # l=0 to maxL then -maxL to l=-1 in the second.
        self._coeffs = np.zeros((maxP+1, 2*maxL+1), dtype=complex)
        self._coeffs[0][0] = complex(1., 0)

        self._k = k    # The wavenumber of the G-L modes
        self._w0 = w0  # The beam waist radius of the G-L modes
        self._maxP = maxP # The highest index of the axial modes included in the modeset
                       # axial mode index p is in the range 0 < p < maxP
        self._maxL = maxL # The highest absolute index of the azimuthal modes included in the modeset
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
        self._coeffs.resize(p+1, self._maxL*2+1, refcheck=False)
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
            newCoeffs = np.zeros((self._maxP+1, l*2+1), dtype=complex)
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
            newCoeffs = np.zeros((self._maxP+1, l*2+1), dtype=complex)
            if l>0:
                newCoeffs[:, :l+2] = self._coeffs[:, :l+2]
                newCoeffs[:, fstColDest:] = self._coeffs[:, fstColSrc:lstColSrc]
            else:
                #special case if we are dropping down to only the fundamental azimuthal mode
                newCoeffs[:,0] = self._coeffs[:,0]
            self._coeffs = newCoeffs
        self._maxL = l


    @property
    def maxP(self):
        """return the maximum absolute index for the axial mode index"""
        return self._maxP

    @maxP.setter
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
        self.resizeModeSet(self._maxP, l)


class GaussLaguerreModeSet(GaussLaguerreModeBase):
    """A class holding a set of Gauss-Laguerre modes, defined in the paraxial limit."""
    def __init__(self, w0=1., k=1., maxP = 0, maxL = 0):
        super(GaussLaguerreModeSet, self).__init__(w0, k, maxP, maxL)

    def field(self, rho, phi, z, p=None, l=None):
        """Return the value of the field at rho, phi, z; either for the sum of all modes (p, l) = None,
        for a specified axial mode p (sum over azimuthal modes), or for a specific (p, l) mode."""
        rhoGrid, phiGrid, zGrid = np.meshgrid(rho, phi, z)
        if p!=None and l!=None:
            # We are after a specific mode
            return self._coeffs[p,l]* Glm.Glm(rhoGrid, phiGrid, zGrid, self.w0, self.k, p=p, l=l)
        elif p!=None and l==None:
            # We are after the sum of all azimuthal modes in an axial mode
            result = np.zeros_like(rhoGrid, dtype=np.complex)
            for l in range(-self.maxL, self.maxL+1):
                result += self.field(rho, phi, z, p=p, l=l)
            return result
        elif p==None and l == None:
            # We are after the sum of all modes.
            result = np.zeros_like(rhoGrid, dtype=np.complex)
            for p in range(0, self.maxP+1):
                result += self.field(rho, phi, z, p=p, l=None)
            return result
        else:
            # Shouldn't get here
            raise ValueError, "GaussLaguerreModeSet.field: must set mode index p if mode index l is set"

    def farField(self, theta, phi, p=None, l=None):
        """Return the value of the far field at theta, phi; either for the sum of all modes (p, l) = None,
        for a specified axial mode p (sum over azimuthal modes), or for a specific (p, l) mode."""
        thetaGrid, phiGrid = np.meshgrid(theta, phi)
        if p!=None and l!=None:
            # We are after a specific mode
            return self._coeffs[p,l]* Glm.Glm_ff(thetaGrid, phiGrid, self.w0, self.k, p=p, l=l)
        elif p!=None and l==None:
            # We are after the sum of all azimuthal modes in an axial mode
            result = np.zeros_like(thetaGrid, dtype=np.complex)
            for l in range(-self.maxL, self.maxL+1):
                result += self.farField(theta, phi, p=p, l=l)
            return result
        elif p==None and l == None:
            # We are after the sum of all modes.
            result = np.zeros_like(thetaGrid, dtype=np.complex)
            for p in range(0, self.maxP+1):
                result += self.farField(theta, phi, p=p, l=None)
            return result
        else:
            # Shouldn't get here
            raise ValueError, "GaussLaguerreModeSet.farField: must set mode index p if mode index l is set"

    def decompose(self, data, rho, phi, w=None, R=None, fix_w=True, fix_R=False):
        """Calculate the mode coefficients and Gaussian beam parameters that best represent the
        field in <data>, and fill self._coeffs. Solves for w, R and hence z and w0 that maximizes
        power in fundamental mode unless w and/or R are fixed.

        Arguments:
            data: numpy array containing complex data to be fitted.
            rho: numpy array containing rho values of points in data.
            phi: numpy array containing phi values of points in data.
            w: float giving initial value for Gaussian beam waist w.
            R: float giving initial value for Gaussian beam radius of phase curvature R.
            fix_w: boolean that fixes value for w, so that it is not solved for.
            fix_R: boolean that fixes value for R, so that it is not solved for.

        Returns:
            tuple containing w and R
        """
        if w is None:
            # We should find the rho point where data amplitude is down by 1/e - we are
            # assuming that the field is roughly the fundamental Gaussian
            # But in the meantime, we'll just assume that the field is across a horn aperture
            w = 0.6435*np.amax(rho)
        if R is None:
            # Assume R is ten times the w value - this won't be too far off if the field
            # is from a horn aperture.
            R = 10*w

        # Normalize data
        self._normalization = np.amax(np.abs(data))
        data = data/self._normalization

        if not fix_W:


        for p in range(self.maxP+1):
            for l in range(-self.maxL, self.maxL+1):
                self._coeffs[p][l] = Glm.nb_Apl(data, rho, phi, w, p, l)

        return w, R



class ModifiedGaussLaguerreModeSet(GaussLaguerreModeBase):
    """A class holding a set of modified Gauss-Laguerre modes, using the definition Tuovinen (1992)"""
    def __init__(self, w0=1., k=1., maxP = 0, maxL = 0):
        super(ModifiedGaussLaguerreModeSet, self).__init__(w0, k, maxP, maxL)

    def field(self, rho, phi, z, p=None, l=None):
        """Return the value of the field at rho, phi, z; either for the sum of all modes (p, l) = None,
        for a specified axial mode p (sum over azimuthal modes), or for a specific (p, l) mode."""
        rhoGrid, phiGrid, zGrid = np.meshgrid(rho, phi, z)
        if p!=None and l!=None:
            # We are after a specific mode
            return self._coeffs[p,l]* modGlm.Glm(rhoGrid, phiGrid, zGrid, self.w0, self.k, p=p, l=l)
        elif p!=None and l==None:
            # We are after the sum of all azimuthal modes in an axial mode
            result = np.zeros_like(rhoGrid, dtype=np.complex)
            for l in range(-self.maxL, self.maxL+1):
                result += self.field(rho, phi, z, p=p, l=l)
            return result
        elif p==None and l == None:
            # We are after the sum of all modes.
            result = np.zeros_like(rhoGrid, dtype=np.complex)
            for p in range(0, self.maxP+1):
                result += self.field(rho, phi, z, p=p, l=None)
            return result
        else:
            # Shouldn't get here
            raise ValueError, "ModifiedGaussLaguerreModeSet.field: must set mode index p if mode index l is set"

    def farField(self, theta, phi, p=None, l=None):
        """Return the value of the far field at theta, phi; either for the sum of all modes (p, l) = None,
        for a specified axial mode p (sum over azimuthal modes), or for a specific (p, l) mode."""
        thetaGrid, phiGrid = np.meshgrid(theta, phi)
        if p!=None and l!=None:
            # We are after a specific mode
            return self._coeffs[p,l] * modGlm.Glm_ff(thetaGrid, phiGrid, self.w0, self.k, p=p, l=l)
        elif p!=None and l==None:
            # We are after the sum of all azimuthal modes in an axial mode
            result = np.zeros_like(thetaGrid, dtype=np.complex)
            for l in range(-self.maxL, self.maxL+1):
                result += self.farField(theta, phi, p=p, l=l)
            return result
        elif p==None and l == None:
            # We are after the sum of all modes.
            result = np.zeros_like(thetaGrid, dtype=np.complex)
            for p in range(0, self.maxP+1):
                result += self.farField(theta, phi, p=p, l=None)
            return result
        else:
            # Shouldn't get here
            raise ValueError, "ModifiedGaussLaguerreModeSet.farField: must set mode index p if mode index l is set"
