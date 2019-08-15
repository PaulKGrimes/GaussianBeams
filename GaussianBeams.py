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

import GaussianLaguerreModes as glm
import ModifiedGaussianLaguerreModes as modGlm

class GaussLaguerreModeBase(object):
    """The base class of the GLM and modified GLM classes.  This base class implements the common parameters and
    handles the storage and manipulation of the mode coefficients"""
    def __init__(self, w0=1., z= 10.0, lm=1., maxP = 0, maxL = 0):
        # Create a complex array holding the mode coefficients of the G-L modes
        # Indexing runs from p=0 to p=maxP in the first dimension and
        # l=0 to maxL then -maxL to l=-1 in the second.
        self.coeffs = np.zeros((maxP+1, 2*maxL+1), dtype=complex)
        self.coeffs[0][0] = complex(1., 0)

        self._lm = lm    # The wavenumber of the G-L modes
        self._w0 = w0  # The beam waist radius of the G-L modes
        self._z = z
        self._maxP = maxP # The highest index of the axial modes included in the modeset
                       # axial mode index p is in the range 0 < p < maxP
        self._maxL = maxL # The highest absolute index of the azimuthal modes included in the modeset
                       # azimuthal mode index l is in the range -maxL < l < maxL

        self.fix_w0 = True # Control whether updating R, z, w updates w0 or z

    @property
    def k(self):
        """return wavenumber k"""
        return 2*np.pi/self._lm

    @k.setter
    def k(self, newK):
        """Set the value of k"""
        self._lm = 2*np.pi/newK

    @property
    def lm(self):
        """Return wavelength lm"""
        return self._lm

    @lm.setter
    def lm(self, newLm):
        self._lm = newLm

    @property
    def w0(self):
        """return the beam waist radius w0"""
        return self._w0

    @w0.setter
    def w0(self, newW):
        """Set the value of w0"""
        self._w0 = newW

    @property
    def z(self):
        """return the distance along the Gaussian beam"""
        return self._z

    @z.setter
    def z(self, newZ):
        """Set the new value of z"""
        self._z = newZ

    @property
    def R(self):
        """return radius of phase curvature of the Gaussian beam"""
        return glm.R_from_w0z(self.w0, self.z, self.lm)

    @R.setter
    def R(self, newR):
        """set the radius of phase curvature of the Gaussian beam"""
        if self.fix_w0:
            self._z = glm.z_from_w0R(self.w0, newR, self.lm)
        else:
            w = self.w
            self._z = glm.z_from_wR(w, newR, self.lm)
            self._w0 = glm.w0_from_wR(w, newR, self.lm)

    @property
    def w(self):
        """return the beam radius at z"""
        return glm.w_from_w0z(self.w0, self.z, self.lm)

    @w.setter
    def w(self, newW):
        """set the radius of the beam at z"""
        if self.fix_w0:
            self._z = glm.z_from_w0w(self.w0, newW, self.lm)
        else:
            R = self.R
            self._z = glm.z_from_wR(newW, R, self.lm)
            self._w0 = glm.w0_from_wR(newW, R, self.lm)

    def resizeModeSet(self, p, l):
        """Resize the array of mode coefficients"""
        # Don't have to do anything clever with p indices
        self.coeffs.resize(p+1, self._maxL*2+1, refcheck=False)
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
            newCoeffs[:, :fstColSrc] = self.coeffs[:, :fstColSrc]
            newCoeffs[:, fstColDest:] = self.coeffs[:, fstColSrc:lstColSrc]
            self.coeffs = newCoeffs
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
                newCoeffs[:, :l+2] = self.coeffs[:, :l+2]
                newCoeffs[:, fstColDest:] = self.coeffs[:, fstColSrc:lstColSrc]
            else:
                #special case if we are dropping down to only the fundamental azimuthal mode
                newCoeffs[:,0] = self.coeffs[:,0]
            self.coeffs = newCoeffs
        self._maxL = l


    @property
    def maxP(self):
        """return the maximum absolute index for the axial mode index"""
        return self._maxP

    @maxP.setter
    def maxP(self, p):
        """Set a new value for maxP"""
        # resize the self.coeffs array
        if p < self.maxL:
            self.maxL = p
        self.resizeModeSet(p, self._maxL)

    @property
    def maxL(self):
        """return the maximum absolute index for the azimuthal mode index"""
        return self._maxL

    @maxL.setter
    def maxL(self, l):
        """Set a new value for maxL"""
        # resize the self.coeffs array
        if l > self.maxP:
            self.maxP = l
        self.resizeModeSet(self._maxP, l)


class GaussLaguerreModeSet(GaussLaguerreModeBase):
    """A class holding a set of Gauss-Laguerre modes, defined in the paraxial limit."""
    def __init__(self, w0=1., k=1., maxP = 0, maxL = 0):
        super(GaussLaguerreModeSet, self).__init__(w0, k, maxP, maxL)

    def field(self, rho, phi, p=None, l=None):
        """Return the value of the field at rho, phi, z; either for the sum of all modes (p, l) = None,
        for a specified axial mode p (sum over azimuthal modes), or for a specific (p, l) mode."""
        rhoGrid, phiGrid = np.meshgrid(rho, phi)
        if p!=None and l!=None:
            if l > p:
                raise RuntimeError("azimuthal mode index l cannot exceed axial mode index p")
            # We are after a specific mode
            return self.coeffs[p,l]* glm.Epl(rhoGrid, phiGrid, self.k, self.w, self.R, p=p, l=l)
        elif p!=None and l==None:
            # We are after the sum of all azimuthal modes in an axial mode
            result = np.zeros_like(rhoGrid, dtype=np.complex)
            # l can only range from -p to +p
            maxL = self.maxL
            if maxL > p:
                maxL = p
            for l in range(-maxL, maxL+1):
                result += self.field(rho, phi, p=p, l=l)
            return result
        elif p==None and l == None:
            # We are after the sum of all modes.
            result = np.zeros_like(rhoGrid, dtype=np.complex)
            for p in range(0, self.maxP+1):
                result += self.field(rho, phi, p=p, l=None)
            return result
        else:
            # Shouldn't get here
            raise ValueError("GaussLaguerreModeSet.field: must set mode index p if mode index l is set")

    def decompose(self, data, rho, phi):
        """Calculate the coefficients that best represent the field in data with
        the G-L mode set.

        Arguments:
            data: numpy array over rho and phi containing the field to fit.
            rho: numpy array of the rho values in data
            phi: numpy array of the phi values in data
        Returns:
            residuals: numpy array of the difference between input data and
            the calculate G-L mode set.
        """
        # Calculate overlap integrals between each mode and the data and store
        # as raw coefficients
        for p in range(0, self.maxP+1):
            maxL = self.maxL
            if maxL > p:
                maxL = p
            for l in range(-maxL, maxL+1):
                self.coeffs[p, l] = self.overlapIntegral(data, rho, phi, p, l)

        # Normalize coefficients to give correct on axis value
        # get coordinates of zero rho and phi
        x = np.argmin(rho)
        y = np.argmin(phi)

        cal_factor = data[x, y]/self.field(rho[x], phi[y])

        self.coeffs = self.coeffs/cal_factor

        # Return residuals
        return data - self.field(rho, phi)

    def overlapIntegral(data, rho, phi, p=0, l=0):
        """Calculate the overlap integral between data and the p, l mode.

        Arguments:
            data: numpy array over rho and phi containing the input field
            rho: numpy array of the rho values in data
            phi: numpy array of the phi values in data
        Returns:
            complex value of the overlap integral between data and p, l mode.
        """
        return 1.0

    # def farField(self, theta, phi, p=None, l=None):
    #     """Return the value of the far field at theta, phi; either for the sum of all modes (p, l) = None,
    #     for a specified axial mode p (sum over azimuthal modes), or for a specific (p, l) mode."""
    #     thetaGrid, phiGrid = np.meshgrid(theta, phi)
    #     if p!=None and l!=None:
    #         # We are after a specific mode
    #         return self.coeffs[p,l]* Glm.Epl_ff(thetaGrid, phiGrid, self.w, self.R, p=p, l=l)
    #     elif p!=None and l==None:
    #         # We are after the sum of all azimuthal modes in an axial mode
    #         result = np.zeros_like(thetaGrid, dtype=np.complex)
    #         for l in range(-self.maxL, self.maxL+1):
    #             result += self.farField(theta, phi, p=p, l=l)
    #         return result
    #     elif p==None and l == None:
    #         # We are after the sum of all modes.
    #         result = np.zeros_like(thetaGrid, dtype=np.complex)
    #         for p in range(0, self.maxP+1):
    #             result += self.farField(theta, phi, p=p, l=None)
    #         return result
    #     else:
    #         # Shouldn't get here
    #         raise ValueError, "GaussLaguerreModeSet.farField: must set mode index p if mode index l is set"



# class ModifiedGaussLaguerreModeSet(GaussLaguerreModeBase):
#     """A class holding a set of modified Gauss-Laguerre modes, using the definition Tuovinen (1992)"""
#     def __init__(self, w0=1., k=1., maxP = 0, maxL = 0):
#         super(ModifiedGaussLaguerreModeSet, self).__init__(w0, k, maxP, maxL)
#
#     def field(self, rho, phi, z, p=None, l=None):
#         """Return the value of the field at rho, phi, z; either for the sum of all modes (p, l) = None,
#         for a specified axial mode p (sum over azimuthal modes), or for a specific (p, l) mode."""
#         rhoGrid, phiGrid, zGrid = np.meshgrid(rho, phi, z)
#         if p!=None and l!=None:
#             # We are after a specific mode
#             return self.coeffs[p,l]* modGlm.Epl(rhoGrid, phiGrid, zGrid, self.w, self.R, p=p, l=l)
#         elif p!=None and l==None:
#             # We are after the sum of all azimuthal modes in an axial mode
#             result = np.zeros_like(rhoGrid, dtype=np.complex)
#             for l in range(-self.maxL, self.maxL+1):
#                 result += self.field(rho, phi, z, p=p, l=l)
#             return result
#         elif p==None and l == None:
#             # We are after the sum of all modes.
#             result = np.zeros_like(rhoGrid, dtype=np.complex)
#             for p in range(0, self.maxP+1):
#                 result += self.field(rho, phi, z, p=p, l=None)
#             return result
#         else:
#             # Shouldn't get here
#             raise ValueError, "ModifiedGaussLaguerreModeSet.field: must set mode index p if mode index l is set"
#
#     def farField(self, theta, phi, p=None, l=None):
#         """Return the value of the far field at theta, phi; either for the sum of all modes (p, l) = None,
#         for a specified axial mode p (sum over azimuthal modes), or for a specific (p, l) mode."""
#         thetaGrid, phiGrid = np.meshgrid(theta, phi)
#         if p!=None and l!=None:
#             # We are after a specific mode
#             return self.coeffs[p,l] * modGlm.Glm_ff(thetaGrid, phiGrid, self.w0, self.k, p=p, l=l)
#         elif p!=None and l==None:
#             # We are after the sum of all azimuthal modes in an axial mode
#             result = np.zeros_like(thetaGrid, dtype=np.complex)
#             for l in range(-self.maxL, self.maxL+1):
#                 result += self.farField(theta, phi, p=p, l=l)
#             return result
#         elif p==None and l == None:
#             # We are after the sum of all modes.
#             result = np.zeros_like(thetaGrid, dtype=np.complex)
#             for p in range(0, self.maxP+1):
#                 result += self.farField(theta, phi, p=p, l=None)
#             return result
#         else:
#             # Shouldn't get here
#             raise ValueError, "ModifiedGaussLaguerreModeSet.farField: must set mode index p if mode index l is set"
