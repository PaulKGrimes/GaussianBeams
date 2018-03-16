    def __init__(self, k=1, w=1, p=0, l=0):
        self.k = k
        self.w = w
        self.p = p
        self.l = l

    def Aint(d, u, w, p=0, l=0):
        """Return the integrand of the expression for the generalized Gaussian-Laguerre mode coefficient of order n, alpha,
        given normalized data d, normalized radial distance u and beam waist W"""
        u2overw2 = u**2 / w**2
        return d*sp.special.eval_genlaguerre(p, l, 2*u2overw2)*np.exp(-u2overw2)*u

    def Apl(d, u, w, p=0, l=0):
        """Return the mode coefficient of the pth radial and lth azimuthal Gaussian-Laguerre mode in d, using a beamwidth w"""
        return (4/(w)**2)*sp.integrate.simps(Aint(d, u, w, p, l), u)

    def Lpl(u, w, p=0, l=0):
        """Return the normalized value at u of the pth radial and lth azimuthal Gaussian-Laguerre mode with a beamwidth w"""
        u2overw2 = u**2 / w**2
        return sp.special.eval_genlaguerre(p, l, 2*u2overw2)*np.exp(-u2overw2)

    def Epl(apl, u, w, p=0, l=0):
        """Return the value at u of the pth radial and lth azimuthal Gaussian-Laguerre mode with a beamwidth w"""
        return apl*Lpl(u, w, p, l)

    def etapl(d, u, apl, w, p=0, l=0):
        """Return the fractional power contained in the pth radial and lth azimuthal Gaussian-Laguerre mode in d, using a beamwidth w"""
        etaTop = sp.integrate.simps((Epl(apl, u, w, p, l)**2)*2*np.pi*u, u)
        etaBot = sp.integrate.simps((d**2)*2*np.pi*u, u)
        return etaTop/etaBot
        

class ModGaussLaguerreMode:
    """A modified Gauss-Laguerre mode, as defined in Tuovinen 1992"""
    def __init__(self, k=1, w0=1, p=0, l=0):
        self.k = k
        self.w0 = w0
        self.p = p
        self.l = l

    def b(kk, w0):
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
        
    def alpha(rho, ww, FF):
        return np.sqrt(2)*rho/(ww*FF)

    def alpha_ff(theta, kk, w0):
        """return the value of alpha"""
        return 1./np.sqrt(2) * kk * w0 * np.sin(theta)

    def Lpl(x, p=0, l=0):
        """return the value of the Gaussian-Laguerre polynomial at x"""
        return sp.special.eval_genlaguerre(p, l, x)

    def Cpl(p=0, l=0):
        """return the normalization factor for the p,lth G-L mode"""
        return np.sqrt(4*np.pi)*np.sqrt((2*factorial(p)/(np.pi*factorial(abs(l)+p))))

    def exponent(rho, phi, z, k, w0, p=0, l=0):
        """return the exponent of e in Glm_ff mode"""
        bb = b(k, w0)
        ww = w(z, w0, bb)
        FF = F(rho, z, bb)
        aa = alpha(rho, ww, FF)
        RR = R(z, bb)
        PP = Phi0(z, bb)
        
        return -((aa**2)/2 - j*k*RR*(FF-1) - j*k*z + j*(2*p + abs(l) + 1)*PP + j*l*phi)
        

    def exponent_ff(theta, phi, a, p=0, l=0):
        """return the exponent of e in Glm_ff mode"""
        theta_d = theta+np.pi*1e-16
        exp = -(a**2)/2 + np.sign(theta)*j*(2*p + abs(l) + 1)*np.pi/2 + j*l*phi
        return exp



    def Glm(rho, phi, z, k, w0, p=0, l=0):
        """return the value of the modified G-L mode"""
        bb = b(k, w0)
        ww = w(z, w0, b)
        FF = F(rho, z, b)
        aa = alpha(rho, ww, FF)
        cosTh = 1.0/FF
        
        return Cpl(p, l) * (1+cosTh)/2 * 1/(k*ww*FF) * aa**abs(l) * Lpl(aa**2)**abs(l) * np.exp(exponent(rho, z, ph, k, w0, p, 1))
        
    def Glm_ff(theta, phi, k, w0, p=0, l=0):
        """return the Farfield value of the modified G-L mode"""
        a = alpha_ff(theta, k, w0)
    return Cpl(p, l)*(1+np.abs(np.cos(theta)))/4*k*w0*a**abs(l)*Lpl(a**2, p, l)**abs(l)*np.exp(exponent_ff(theta, phi, a, p, l))