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

from scipy.integrate import simps

import GaussianLaguerreModes as glm
import ModifiedGaussianLaguerreModes as modGlm

j = complex(0, 1)

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
        self._z = z     # Distance along the beam
        self._phase = 0.0  # An arbitrary phase factor
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
    def phase(self):
        """Return the arbitrary phase"""
        return self._phase

    @phase.setter
    def phase(self, newPhase):
        """Set an arbitrary phase"""
        self._phase = newPhase

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

    @property
    def phi0(self):
        """Return the beam phase factor.

        Leave this as read only, as there's no real reason to set it directly"""
        return glm.phi0_from_w0z(self.w0, self.z, self.lm)

    def resizeModeSet(self, p, l):
        """Resize the array of mode coefficients to p, 2*l+1 modes. Modes with
        l>p are undefined, but values are checked by maxL and maxP setters, not
        here.

        Arguments:
            p: integer number of axial modes to include.
            l: integer maximum azimuthal mode number to include. Number of azimuthal
                modes is 2*l+1."""
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
        for a specified axial mode p (sum over azimuthal modes), or for a specific (p, l) mode.

        Arguments:
            rho: numpy array of the rho values.
            phi: numpy array of the phi values.
            p=None: axial mode number - integer.
            l=None: azimuth mode number - integer.
        Returns:
            numpy array over a meshgrid of rho, phi containing complex
            field values.
        """
        rhoGrid, phiGrid = np.meshgrid(rho, phi)
        if p!=None and l!=None:
            if l > p:
                raise RuntimeError("azimuthal mode index l cannot exceed axial mode index p")
            # We are after a specific mode
            return self.coeffs[p,l]* glm.Epl(rhoGrid, phiGrid, self.k, self.w, self.R, p=p, l=l)*np.exp(j*self.phase)
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
                self.coeffs[p, l] = self.modeOverlapIntegral(data, rho, phi, p, l)

        # Normalize coefficients to give correct on axis value
        # get coordinates of zero rho and phi
        #x = np.argmin(np.abs(rho))
        #y = np.argmin(np.abs(phi))

        #cal_factor = data[x, y]/self.field(rho[x], phi[y])

        #self.coeffs = self.coeffs*cal_factor

        # Return residuals
        return data - self.field(rho, phi)

    def modeOverlapIntegral(self, data, rho, phi, p=0, l=0):
        """Calculate the overlap integral between data and the normalized p, l
        mode.

        Arguments:
            data: numpy array over rho and phi containing the input field
            rho: numpy array of the rho values in data
            phi: numpy array of the phi values in data
            p=0: axial mode number to include in overlap integral
            l=0: azimuthal mode number to include in overlap integral
        Returns:
            complex value of the overlap integral between data and p, l mode.
        """
        return glm.Apl(data, rho, phi, self.k, self.w, self.R, p, l)


    def overlapIntegral(self, data, rho, phi, p=None, l=None):
        """Calculate the overlap integral between data and modeset, either for
        the sum of all modes (p, l) = None, for a specified axial mode p (sum
        over azimuthal modes), or for a specific (p, l) mode.  Coefficients will
        be included.

        Can pad data with zeros to extend beyond given dataset. This is useful
        when data is given over the aperture of a horn or stop, and we want to
        constrain the fitted modeset to be zero outside the aperture.

        Arguments:
            data: numpy array over rho and phi containing the input field
            rho: numpy array of the rho values in data
            phi: numpy array of the phi values in data
            p=None: axial mode number to include in overlap integral
            l=None: azimuthal mode number to include in overlap integral
        Returns:
            complex value of the overlap integral between data and p, l mode.
        """
        field = self.field(rho, phi, p, l)

        integrand = data*np.conj(field)*np.abs(rho)

        overlap = simps(simps(integrand, phi, axis=0, even="avg"), rho, even="first")

        return overlap


    def powerIntegral(self, rho, phi, p=None, l=None, padRho=None):
        """Calculate the integrated power in the field; either for the sum of
        all modes (p, l) = None, for a specified axial mode p (sum over
        azimuthal modes), or for a specific (p, l) mode.

        Can pad the rho vector to allow calculation beyond the rho vector
        supplied, in order to make sure that all power is included.

        Arguments:
            rho: numpy array of the rho values.
            phi: numpy array of the phi values.
            p=None: axial mode number - integer.
            l=None: azimuth mode number - integer.
        Returns:
            Power in the field - float.
        """
        if padRho:
            # Assume that rho is an ordered and evenly spaced set
            rho = np.linspace(rho[0]*padRho, rho[-1]*padRho, len(rho)*padRho)

        field = self.field(rho, phi, p, l)
        integrand = field*np.conj(field)*np.abs(rho)

        power = simps(simps(integrand, phi, axis=0, even="avg"), rho, even="first")
        norm = simps(simps(np.ones_like(integrand)*np.abs(rho), phi, axis=0, even="avg"), rho, even="first")

        return power/norm

    def eta_pl(self, rho, phi, p=0, l=0):
        """Calculate the fraction of integrated power in the field present in
        mode p, l.

        Arguments:
            rho: numpy array of the rho values.
            phi: numpy array of the phi values.
            p=0: axial mode number - integer.
            l=0: azimuth mode number - integer.
        Returns:
            Power in the field - float.
        """
        return np.abs(self.powerIntegral(rho, phi, p, l)/self.powerIntegral(rho, phi))

    def fit_func(self, data, rho, phi, w, R):
        """Return a function that represents the negative of the fractional
        power in the p=0,l=0 mode, when fitted over data, rho, phi.

        Arguments:
            data: numpy array over rho and phi containing the input field
            rho: numpy array over the rho values in data.
            phi: numpy array over the phi values in data.

        Returns:
            fun: function of r, W
        """
        self.w = w
        self.R = R

        self.decompose(data, rho, phi)

        return -self.eta_pl(rho, phi, p=0, l=0)

    def fit_w_R(self, data, rho, phi):
        """Calculate the w and R that maximises the power in the fundamental
        Gauss-Laguerre mode.

        Arguments:
            data: numpy array over rho and phi containing the input field
            rho: numpy array over the rho values in data.
            phi: numpy array over the phi values in data.

        Returns:
            res: Results from scipy.minimize
        """
        # Make sure that setting R and w changes w0
        fix_w0 = self.fix_w0
        self.fix_w0 = False

        old_w = self.w
        old_R = self.R

        fun = lambda x : self.fit_func(data, rho, phi, x[0], x[1])

        if self.z > 0:
            Rbound = (0.0, None)
            wbound = (self.lm, None)

        res = sp.optimize.minimize(fun, (self.w, self.R), method='L-BFGS-B', bounds=(wbound, Rbound))

        if res.success:
            self.w = res.x[0]
            self.R = res.x[1]
        else:
            self.w = old_w
            self.R = old_R
            print("Optimization failed with message: {:s}".format(res.message))

        self.fix_w0 = fix_w0

        return res

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
