"""
This file contains the PPmodel class. It aims at computing
the gamma, electron and neutrinos spectrum for hadronic interactions between 
cosmic rays and ICM. It is based on Kelner et al. 2006, and compute spectra 
given as dN/dEdVdt in GeV-1 cm-3 s-1.
It is inspired by the Naima package.

"""

#==================================================
# Requested imports
#==================================================

import numpy as np
from astropy import constants as const
import astropy.units as u
from scipy.integrate import quad
import scipy.integrate as integrate

#==================================================
# Class
#==================================================

class PPmodel(object):
    """ Observable class
    This class compute gamma, electron and neutrinos spectra
    for hadronic interaction p_gas + p_CR.
    Attributes
    ----------  
    - Jp (function): a function which take the proton energy (in GeV) as input
    and returns the number of cosmic ray proton in unit of GeV-1 cm-3. 
    Jp can also depend on radius (in kpc), in which case the radius should be provided 
    when calling the spectrum. Jp is defined as Jp(radius, energy)[i_radius, j_energy].
    - Epmin (quantity): minimal proton energy
    - Epmax (quantity): maximum proton energy
    - NptEpPD (int): number of proton energy per decade
    - _m_pi (float): the pion mass in GeV
    - _m_p (float): the proton mass in GeV
    - _Kpi (float): the pion average multiplicity
    - _Etrans (float): the transition energy between delta approximation and Kelner 
    scaling (in GeV)
    - _norm (float): the normalization of the spectrum
    Methods
    ----------  
    - gamma_spectrum(self, Egamma_input, limit='mixed'): compute the gamma ray spectrum
    - electron_spectrum(self, Ee_input, limit='mixed'): compute the electron spectrum
    - neutrino_spectrum(self, Enu_input, limit='mixed', flavor='numu'): compute the neutrino spectrum
    - get_cr_energy_density(self, Emin=None, Emax=None): compute the energy stored in the CR protons
    - set_cr_energy_density(self, Ucr, Emin=None, Emax=None): set the energy stored in the CR protons
    """
    
    #========== Init
    def __init__(self, Jp,
                 Epmin=None,
                 Epmax=None,
                 NptEpPd=100):

        self._m_pi = 0.1349766
        self._m_p = (const.m_p*const.c**2).to_value('GeV')
        self._Kpi = 0.17
        self._Etrans = 100.0
        self._norm = 1.0

        self.Jp = Jp
        if Epmin is None:
            Epmin = self._m_p + 2*self._m_pi + self._m_pi**2 / 2.0 / self._m_p
        else:
            Epmin = Epmin.to_value('GeV')
        if Epmax is None:
            Epmax = 1e7
        else:
            Epmax = Epmax.to_value('GeV')
        self._Epmin = Epmin
        self._Epmax = Epmax
        self._NptEpPd = NptEpPd
        self.Ep = np.logspace(np.log10(Epmin), np.log10(Epmax), int(NptEpPd*(np.log10(Epmax/Epmin))))
        
    #========== Heaviside function
    def _heaviside(self, x):
        """
        Compute the heaviside fonction for a scalar or vector
        Parameters
        ----------
        - x: scalar or vector
        
        Outputs
        --------
        - heaviside(x)
        """
        return (np.sign(x) + 1) / 2.0

    #========== Jp*Ep
    def _JpEp(self, Ep, radius=None):
        """ 
        Multiply the proton distribution by the energy
        
        Parameters
        ----------
        - Ep (GeV) : the proton energy
        - radius (kpc): the radius in case Jp is a 2d function
        
        Outputs
        --------    
        - Ep Jp(Ep)
        """

        if radius is None:
            func = Ep * self.Jp(Ep)
        else:
            func = Ep*self.Jp(radius, Ep) #Ep and Jp have different dim, but this is ok with python
            
        return func

    
    #========== pp interactioin kinematic threshold
    def _pp_pion_kinematic_energy_threshold(self):
        """
        Compute the kinematic energy threshold of pp collision -> pions
        
        Parameters
        ----------
        
        Outputs
        --------
        - Energy threshold (GeV)
        """
        return self._m_p + 2*self._m_pi + self._m_pi**2 / 2.0 / self._m_p # in unit of m_p, m_pi

    #========== Cross section
    def _sigma_inel(self, Ep):
        """
        Give the proton proton interaction cross-section as 
        a function of energy.
        
        Parameters
        ----------
        - E_p : the energy of protons in GeV
        
        Outputs
        --------
        - sig_pp : the cross section in cm2
        """

        Eth = self._pp_pion_kinematic_energy_threshold()

        L = np.log(Ep/1000.0)
        sigma = 34.3 + 1.88 * L + 0.25 * L ** 2
        
        #if Ep <= 0.1*1e3:
        #    sigma *= (1 - (Eth / Ep) ** 4) ** 2 * self._heaviside(Ep - Eth)
        sigma *= (1 - (Eth / Ep) ** 4) ** 2 * self._heaviside(Ep - Eth)
        
        return sigma * 1e-27
    
    #========== Fgamma term
    def _Fgamma(self, x, Ep):
        """
        Compute the term from Eq.58 of Kelner et al. 2006
        
        Parameters
        ----------
        - x = E_gamma/E_proton
        - Ep = E_proton (GeV)
        
        Outputs
        --------
        - The F_gamma term
        """
    
        L = np.log(Ep/1000.0)
        B = 1.30 + 0.14 * L + 0.011 * L ** 2
        beta = (1.79 + 0.11 * L + 0.008 * L ** 2) ** -1
        k = (0.801 + 0.049 * L + 0.014 * L ** 2) ** -1
        xb = x ** beta
        
        F1 = B * (np.log(x) / x) * ((1 - xb) / (1 + k * xb * (1 - xb))) ** 4
        F2 = (1.0 / np.log(x)
              - (4 * beta * xb) / (1 - xb)
              - (4 * k * beta * xb * (1 - 2 * xb)) / (1 + k * xb * (1 - xb))
        )
        
        return F1 * F2 # unitless

    #========== Fe term
    def _Fe(self, x, Ep):
        """
        Compute the term from Eq.62 of Kelner et al. 2006. This is also the
        same for nu_e (within 5%), and F_nu_mu(2)
        
        Parameters
        ----------
        - x = E_e/E_proton, or E_nu/E_proton
        - Ep = E_proton (GeV)
        
        Outputs
        --------
        - The F_e term
        """
    
        L = np.log(Ep/1000.0)
        
        B_e = (69.5 + 2.65*L + 0.3*L**2) ** -1
        beta_e = (0.201 + 0.062*L + 0.00042*L**2) ** (-1.0/4.0)
        k_e = (0.279 + 0.141*L + 0.0172*L** 2) / (0.3 + (2.3 + L)**2)

        F1 = (1 + k_e * (np.log(x))**2)**3
        F2 = (x * (1 + 0.3/x**beta_e))**-1
        F3 = (-np.log(x))**5
        
        F = B_e * F1 *F2 * F3

        return F # unitless

    #========== Fe term
    def _Fnumu1(self, x, Ep):
        """
        Compute the term from Eq.66 of Kelner et al. 2006.
        
        Parameters
        ----------
        - x = E_nu_mu/E_proton
        - Ep = E_proton (GeV)
        
        Outputs
        --------
        - The F_nu_mu term
        """
    
        L = np.log(Ep/1000.0)
        y = x/0.427
        
        Bprime     = 1.75 + 0.204*L + 0.010*L**2
        beta_prime = (1.67 + 0.111*L + 0.0038*L**2)**-1
        k_prime    = 1.07 - 0.086*L + 0.002*L**2
        
        F1 = np.log(y) / y
        F2 = ((1 - y**beta_prime) / (1 + k_prime * y**beta_prime * (1 - y**beta_prime)))**4
        F3a = 1.0/np.log(y)
        F3b = (4.0*beta_prime*y**beta_prime) / (1-y**beta_prime)
        F3c = (4.0*k_prime*beta_prime*y**beta_prime * (1.0 - 2.0*y**beta_prime))/(1.0 + k_prime*y**beta_prime*(1.0-y**beta_prime))
        
        F = Bprime * F1 * F2 * (F3a - F3b - F3c) * self._heaviside(0.427-x)

        return F # unitless
    
    #========== High energy gamma integrand of function
    def _highE_integrand(self, x, Energy, case, radius=None):
        """
        Defines the integrand of eq 72 from Kelner et al. (2006).
        
        Parameters
        ----------
        - x = E_{gamma,electron,neutrino} / E_proton (unitless)
        - Energy = Energy of the gamma ray or electron, or neutrino (GeV)
        - case: which spectrum, 'gamma', 'electron', 'numu' or 'nue'
        
        Outputs
        --------
        - The integrand in units of cm-1 Gev^-1
        """
        
        try:
            if case == 'gamma':
                Ffunc =  self._Fgamma(x, Energy/x)
            elif case == 'electron':
                Ffunc =  self._Fe(x, Energy/x)
            elif case == 'nue':
                Ffunc = self._Fe(x, Energy/x)
            elif case == 'numu':
                Ffunc = self._Fnumu1(x, Energy/x) + self._Fe(x, Energy/x)
            else:
                raise ValueError("Only 'gamma', 'electron', 'numu' or 'nue' are available")

            if radius is None:
                Jpfunc = self.Jp(Energy/x)
            else:
                Jpfunc = self.Jp(radius, Energy/x)
            
            return (self._sigma_inel(Energy/x)* Jpfunc * Ffunc/x)
            
        except ZeroDivisionError:
            return np.nan

    #========== Compute the gamma spectrum for high E limit
    def _calc_specpp_hiE(self, Energy, case, radius=None):
        """
        Compute the spectrum folowing Kelner 2006, i.e. in the high energy regime.
        
        Parameters
        ----------
        - Energy = the gamma ray or electron energy vector in GeV
        - case: which spectrum, 'gamma', 'electron', 'numu' or 'nue'
        Outputs
        --------
        - The normalized photon count in unit of GeV-1 cm-1
        """

        #specpp = quad(self._highE_integrand, 0.0, 1.0, args=(Energy, case), epsrel=1e-3,epsabs=0)[0]

        xmin = Energy/self._Epmax
        xmax = np.amin([Energy/self._Epmin, 1.0-1e-10])

        if xmin < xmax:
            x = np.logspace(np.log10(xmin), np.log10(xmax), int(self._NptEpPd*(np.log10(xmax/xmin))))
        
            if radius is None:
                specpp = self._trapz_loglog(self._highE_integrand(x, Energy, case), x)
            else:
                specpp = self._trapz_loglog(self._highE_integrand(x, Energy, case, radius), x, axis=1)
        else:
            if radius is None:
                specpp = 0
            else:
                specpp = radius*0

        return specpp
    
    #========== Integrand for low E limit
    def _delta_integrand(self, Epi, radius=None):
        """
        Defines the integrand of eq 78 from Kelner et al. (2006) in the 
        low energy limit following the delta approximation.
        
        Parameters
        ----------
        - Epi = energy of pions (GeV)
        
        Outputs
        --------
        - The integrand in units of Gev^-2 cm^-1
        """
        
        Ep0 = self._m_p + Epi / self._Kpi
        if radius is None:
            func = self._sigma_inel(Ep0)* self.Jp(Ep0) / np.sqrt(Epi**2 - self._m_pi**2)
        else:
            func = self._sigma_inel(Ep0)* self.Jp(radius, Ep0) / np.sqrt(Epi**2 - self._m_pi**2)

        return func
        
    #========== Compute the spectrum with delta approximation
    def _calc_specpp_loE(self, Energy, radius=None, ntilde=1):
        """
        Compute the spectrum in the low energy regime. The normalization
        here is not important because it is rescaled to the high energy
        at E_lim afterwards.
        
        Parameters
        ----------
        - Energy = the gamma ray or electron energy vector in GeV
        - ntilde = a scaling normalization cose to one
        
        Outputs
        --------    
        - The normalized photon count in GeV-1 cm-1
        """
        
        #Epimin = Energy + self._m_pi**2 / (4 * Energy)
        #result = 2*(ntilde/self._Kpi) * quad(self._delta_integrand, Epimin, np.inf, epsrel=1e-3, epsabs=0)[0]
        
        Emin = Energy + self._m_pi**2 / (4 * Energy)
        Emax = self._Epmax
        NptPd = self._NptEpPd
        E_arr = np.logspace(np.log10(Emin), np.log10(Emax), int(NptPd*(np.log10(Emax/Emin))))
        
        if radius is None:
            result = 2*(ntilde/self._Kpi) * self._trapz_loglog(self._delta_integrand(E_arr), E_arr)
        else:
            result = 2*(ntilde/self._Kpi) * self._trapz_loglog(self._delta_integrand(E_arr, radius), E_arr, axis=1)
        
        return result # s-1 GeV-1

    #========== Compute integral in loglog
    def _trapz_loglog(self, y, x, axis=-1, intervals=False):
        """
        Integrate along the given axis using the composite trapezoidal rule in
        loglog space. Integrate y(x) along given axis in loglog space. This follows 
        the script in the Naima package.
        
        Parameters
        ----------
        - y (array_like): Input array to integrate.
        - x (array_like):  optional. Independent variable to integrate over.
        - axis (int): Specify the axis.
        - intervals (bool): Return array of shape x not the total integral, default: False
        
        Returns
        -------
        trapz (float): Definite integral as approximated by trapezoidal rule in loglog space.
        """
        
        log10 = np.log10

        #----- Check for units
        try:
            y_unit = y.unit
            y = y.value
        except AttributeError:
            y_unit = 1.0
        try:
            x_unit = x.unit
            x = x.value
        except AttributeError:
            x_unit = 1.0

        y = np.asanyarray(y)
        x = np.asanyarray(x)

        #----- Define the slices
        slice1 = [slice(None)] * y.ndim
        slice2 = [slice(None)] * y.ndim
        slice1[axis] = slice(None, -1)
        slice2[axis] = slice(1, None)
        slice1, slice2 = tuple(slice1), tuple(slice2)

        #----- arrays with uncertainties contain objects, remove tiny elements
        if y.dtype == "O":
            from uncertainties.unumpy import log10
            # uncertainties.unumpy.log10 can't deal with tiny values see
            # https://github.com/gammapy/gammapy/issues/687, so we filter out the values
            # here. As the values are so small it doesn't affect the final result.
            # the sqrt is taken to create a margin, because of the later division
            # y[slice2] / y[slice1]
            valid = y > np.sqrt(np.finfo(float).tiny)
            x, y = x[valid], y[valid]

        #----- reshaping x
        if x.ndim == 1:
            shape = [1] * y.ndim
            shape[axis] = x.shape[0]
            x = x.reshape(shape)

        #-----
        with np.errstate(invalid="ignore", divide="ignore"):
            # Compute the power law indices in each integration bin
            b = log10(y[slice2] / y[slice1]) / log10(x[slice2] / x[slice1])

            # if local powerlaw index is -1, use \int 1/x = log(x); otherwise use normal
            # powerlaw integration
            trapzs = np.where(np.abs(b + 1.0) > 1e-10,
                              (y[slice1] * (x[slice2] * (x[slice2] / x[slice1]) ** b - x[slice1]))
                              / (b + 1),
                              x[slice1] * y[slice1] * np.log(x[slice2] / x[slice1]))
            
        tozero = (y[slice1] == 0.0) + (y[slice2] == 0.0) + (x[slice1] == x[slice2])
        trapzs[tozero] = 0.0

        if intervals:
            return trapzs * x_unit * y_unit

        ret = np.add.reduce(trapzs, axis) * x_unit * y_unit

        return ret

    
    #========== Apply normalization
    def _apply_normalization(self, Egamma, spec0, nH):
        """
        Apply the normalization to the spectrum
        
        Parameters
        ----------
        - spec0 (array) : the input spectrum
        Outputs
        --------    
        - spec (array): the normalized spectrum
        """

        norm = const.c.to_value('cm/s') * nH.to_value('cm-3') * self._norm
        norm_grid = (np.tile(norm, [len(Egamma),1])).T
        spec = spec0 * norm_grid
        
        return spec

    #========== Compute the spectrum
    def gamma_spectrum(self, Egamma_input, radius_input=None, nH=1.0*u.cm**-3, limit='mixed'):
        """
        Compute the gamma ray spectrum merging low energy and high energy
        
        Parameters
        ----------
        - Egamma_input (quantity) : the gamma ray energy vector in unit of GeV
        - radius_input (quantity) : array of radius to be  provided if Jp is 2D
        - limit (str): use this keyword to chose only high energy or low energy limits
        The keywords are: 'mixed', 'lowE', 'highE'
        Outputs
        --------    
        - The photon spectrum in unit homogeneous to GeV-1 cm-3 s-1
        """
        
        Egamma = Egamma_input.to_value('GeV')
        if type(Egamma) == float: Egamma = np.array([Egamma])

        #---------- Case of no radius
        if radius_input is None:
            full = self._calc_specpp_hiE(self._Etrans, 'gamma')
            delta = self._calc_specpp_loE(self._Etrans, ntilde=1.0)
            if full != 0 and delta != 0:
                nhat = (full / delta)
            else:
                nhat = 0.0
        
            spec = np.zeros(len(Egamma))
            norm = const.c.to_value('cm/s') * nH.to_value('cm-3') * self._norm

            #----- Standard case: mixed highE + lowE
            if limit == 'mixed':
                for i in range(len(Egamma)): 
                    if Egamma[i] >= self._Etrans:
                        spec[i] = norm*self._calc_specpp_hiE(Egamma[i], 'gamma')
                    else:
                        spec[i] = norm*self._calc_specpp_loE(Egamma[i], ntilde=nhat)

            #----- highE only
            elif limit == 'highE':
                for i in range(len(Egamma)):
                    spec[i] = norm*self._calc_specpp_hiE(Egamma[i], 'gamma')

            #----- lowE only
            elif limit == 'lowE':
                for i in range(len(Egamma)):
                    spec[i] = norm*self._calc_specpp_loE(Egamma[i], ntilde=nhat)

            #----- Error otherwise
            else:
                raise ValueError("Only 'mixed', 'highE', or 'lowE' are available")

        #---------- Case of radius
        else:
            if len(nH) != len(radius_input):
                raise ValueError('nH should have the same size as radius_input')
            
            radius = radius_input.to_value('kpc')

            full = self._calc_specpp_hiE(self._Etrans, 'gamma', radius=radius)
            delta = self._calc_specpp_loE(self._Etrans, radius=radius, ntilde=1.0)
            w0 = (full == 0)*(delta == 0) # Search location of 0 density
            delta[w0] = 1.0 # Avoid dividing by 0
            nhat = (full / delta)
        
            spec0 = np.zeros((len(radius), len(Egamma)))

            #----- Standard case: mixed highE + lowE
            if limit == 'mixed':
                for i in range(len(Egamma)):
                    if Egamma[i] >= self._Etrans:
                        spec0[:,i] = self._calc_specpp_hiE(Egamma[i], 'gamma', radius=radius)
                    else:
                        spec0[:,i] = self._calc_specpp_loE(Egamma[i], radius=radius, ntilde=nhat)

            #----- highE only
            elif limit == 'highE':
                for i in range(len(Egamma)):
                    spec0[:,i] = self._calc_specpp_hiE(Egamma[i], 'gamma', radius=radius)

            #----- lowE only
            elif limit == 'lowE':
                for i in range(len(Egamma)):
                    spec0[:,i] = self._calc_specpp_loE(Egamma[i], radius=radius, ntilde=nhat)

            #----- Error otherwise
            else:
                raise ValueError("Only 'mixed', 'highE', or 'lowE' are available")
        
            #----- Normalization
            spec = self._apply_normalization(Egamma, spec0, nH)
        
        return spec * u.GeV**(-1) * u.cm**(-3) * u.s**(-1)
    
    #========== Compute the spectrum
    def electron_spectrum(self, Ee_input, radius_input=None, nH=1.0*u.cm**-3, limit='mixed'):
        """
        Compute the electron spectrum merging low energy and high energy
        
        Parameters
        ----------
        - Ee_input (quantity) : the electron energy vector in unit of GeV
        - radius_input (quantity) : array of radius to be  provided if Jp is 2D
        - limit (str): use this keyword to chose only high energy or low energy limits
        The keywords are: 'mixed', 'lowE', 'highE'
        Outputs
        --------    
        - The electron spectrum in unit homogeneous to GeV-1 cm-3 s-1
        """
        
        Ee = Ee_input.to_value('GeV')
        if type(Ee) == float: Ee = np.array([Ee])

        Emin_elec = (const.m_e*const.c**2).to_value('GeV')
        
        #---------- Case of no radius
        if radius_input is None:
            full = self._calc_specpp_hiE(self._Etrans, 'electron')
            delta = self._calc_specpp_loE(self._Etrans, ntilde=1.0)
            if full != 0 and delta != 0:
                nhat = (full / delta)
            else:
                nhat = 0.0
                
            spec = np.zeros(len(Ee))
            norm = const.c.to_value('cm/s') * nH.to_value('cm-3') * self._norm

            #----- Standard case: mixed highE + lowE
            if limit == 'mixed':
                for i in range(len(Ee)):
                    if Ee[i] >= Emin_elec:
                        if Ee[i] >= self._Etrans:
                            spec[i] = norm*self._calc_specpp_hiE(Ee[i], 'electron')
                        else:
                            spec[i] = norm*self._calc_specpp_loE(Ee[i], ntilde=nhat)

            #----- highE only
            elif limit == 'highE':
                for i in range(len(Ee)):
                    if Ee[i] >= Emin_elec:
                        spec[i] = norm*self._calc_specpp_hiE(Ee[i], 'electron')

            #----- lowE only
            elif limit == 'lowE':
                for i in range(len(Ee)):
                    if Ee[i] >= Emin_elec:
                        spec[i] = norm*self._calc_specpp_loE(Ee[i], ntilde=nhat)

            #----- Error otherwise
            else:
                raise ValueError("Only 'mixed', 'highE', or 'lowE' are available")

        #---------- Case of radius
        else:
            if len(nH) != len(radius_input):
                raise ValueError('nH should have the same size as radius_input')
            
            radius = radius_input.to_value('kpc')

            full = self._calc_specpp_hiE(self._Etrans, 'electron', radius=radius)
            delta = self._calc_specpp_loE(self._Etrans, radius=radius, ntilde=1.0)
            w0 = (full == 0)*(delta == 0) # Search location of 0 density
            delta[w0] = 1.0 # Avoid dividing by 0
            nhat = (full / delta)
            
            spec0 = np.zeros((len(radius), len(Ee)))

            #----- Standard case: mixed highE + lowE
            if limit == 'mixed':
                for i in range(len(Ee)):
                    if Ee[i] >= Emin_elec:
                        if Ee[i] >= self._Etrans:
                            spec0[:,i] = self._calc_specpp_hiE(Ee[i], 'electron', radius=radius)
                        else:
                            spec0[:,i] = self._calc_specpp_loE(Ee[i], radius=radius, ntilde=nhat)

            #----- highE only
            elif limit == 'highE':
                for i in range(len(Ee)):
                    if Ee[i] >= Emin_elec:
                        spec0[:,i] = self._calc_specpp_hiE(Ee[i], 'electron', radius=radius)

            #----- lowE only
            elif limit == 'lowE':
                for i in range(len(Ee)):
                    if Ee[i] >= Emin_elec:
                        spec0[:,i] = self._calc_specpp_loE(Ee[i], radius=radius, ntilde=nhat)

            #----- Error otherwise
            else:
                raise ValueError("Only 'mixed', 'highE', or 'lowE' are available")
        
            #----- Normalization
            spec = self._apply_normalization(Ee, spec0, nH)
                
        return spec * u.GeV**(-1) * u.cm**(-3) * u.s**(-1)


    #========== Compute the spectrum
    def neutrino_spectrum(self, Enu_input, radius_input=None, nH=1.0*u.cm**-3, limit='mixed', flavor='numu'):
        """
        Compute the neutrino spectrum merging low energy and high energy
        
        Parameters
        ----------
        - Enu_input (quantity) : the neutrinos energy vector in unit of GeV
        - radius_input (quantity) : array of radius to be  provided if Jp is 2D
        - limit (str): use this keyword to chose only high energy or low energy limits
        The keywords are: 'mixed', 'lowE', 'highE'
        - flavor (str): 'numu' or 'nue', which neutrino flavor you want
        Outputs
        --------    
        - The neutrino spectrum in unit homogeneous to GeV-1 cm-3 s-1
        """

        Enu = Enu_input.to_value('GeV')
        if type(Enu) == float: Enu = np.array([Enu])
                
        #---------- Case of no radius
        if radius_input is None:
            full = self._calc_specpp_hiE(self._Etrans, flavor)
            delta = self._calc_specpp_loE(self._Etrans, ntilde=1.0)
            if full != 0 and delta != 0:
                nhat = (full / delta)
            else:
                nhat = 0.0
                
            spec = np.zeros(len(Enu))
            norm = const.c.to_value('cm/s') * nH.to_value('cm-3') * self._norm

            #----- Standard case: mixed highE + lowE
            if limit == 'mixed':
                for i in range(len(Enu)):
                    if Enu[i] >= self._Etrans:
                        spec[i] = norm*self._calc_specpp_hiE(Enu[i], flavor)
                    else:
                        spec[i] = norm*self._calc_specpp_loE(Enu[i], ntilde=nhat)

            #----- highE only
            elif limit == 'highE':
                for i in range(len(Enu)):
                    spec[i] = norm*self._calc_specpp_hiE(Enu[i], flavor)

            #----- lowE only
            elif limit == 'lowE':
                for i in range(len(Enu)):
                    spec[i] = norm*self._calc_specpp_loE(Enu[i], ntilde=nhat)

            #----- Error otherwise
            else:
                raise ValueError("Only 'mixed', 'highE', or 'lowE' are available")

        #---------- Case of radius
        else:
            if len(nH) != len(radius_input):
                raise ValueError('nH should have the same size as radius_input')
            
            radius = radius_input.to_value('kpc')

            full = self._calc_specpp_hiE(self._Etrans, flavor, radius=radius)
            delta = self._calc_specpp_loE(self._Etrans, radius=radius, ntilde=1.0)
            w0 = (full == 0)*(delta == 0) # Search location of 0 density
            delta[w0] = 1.0 # Avoid dividing by 0
            nhat = (full / delta)
            
            spec0 = np.zeros((len(radius), len(Enu)))

            #----- Standard case: mixed highE + lowE
            if limit == 'mixed':
                for i in range(len(Enu)):
                    if Enu[i] >= self._Etrans:
                        spec0[:,i] = self._calc_specpp_hiE(Enu[i], flavor, radius=radius)
                    else:
                        spec0[:,i] = self._calc_specpp_loE(Enu[i], radius=radius, ntilde=nhat)

            #----- highE only
            elif limit == 'highE':
                for i in range(len(Enu)):
                    spec0[:,i] = self._calc_specpp_hiE(Enu[i], flavor, radius=radius)

            #----- lowE only
            elif limit == 'lowE':
                for i in range(len(Enu)):
                    spec0[:,i] = self._calc_specpp_loE(Enu[i], radius=radius, ntilde=nhat)

            #----- Error otherwise
            else:
                raise ValueError("Only 'mixed', 'highE', or 'lowE' are available")
        
            #----- Normalization
            spec = self._apply_normalization(Enu, spec0, nH)
                
        return spec * u.GeV**(-1) * u.cm**(-3) * u.s**(-1)
    
    #========== Get the total energy in CR
    def get_cr_energy_density(self, radius=None, Emin=None, Emax=None):
        """ 
        Compute the total energy of cosmic ray proton
        
        Parameters
        ----------
        - radius (quantity): the radius in case Jp is a 2d function
        - Emin (quantity) : the minimal energy for integration
        - Emax (quantity) : the maximum integration. This is important for 
        slopes <~ 2. From Pinzke et al 2010, we expect a cutoff at E~10^10 GeV.
        
        Outputs
        --------    
        - U_CR = the energy in GeV/cm^-3
        """
        
        #----- Def Emin and Emax
        if Emin is None:
            Emin_lim = self._Epmin
        else:
            Emin_lim = Emin.to_value('GeV')
        
        if Emax is None:
            Emax_lim = self._Epmax
        else:
            Emax_lim = Emax.to_value('GeV')

        Ep = np.logspace(np.log10(Emin_lim), np.log10(Emax_lim), int(self._NptEpPd*(np.log10(Emax_lim/Emin_lim))))

        #----- Integration        
        if radius is None:
            U_CR = self._norm * self._trapz_loglog(self._JpEp(Ep), Ep) # GeV/cm^-3
        else:
            U_CR = self._norm * self._trapz_loglog(self._JpEp(Ep, radius=radius.to_value('kpc')), Ep, axis=1) # GeV/cm^-3 = f(r)

        #U_CR = self._norm * integrate.quad(self._JpEp, Emin_lim, Emax_lim)[0] # Useful for Emax = np.inf

        return U_CR * u.GeV / u.cm**3
    
    #========== Get the total energy in CR
    def set_cr_energy_density(self, Ucr, radius=None, Emin=None, Emax=None):
        """ 
        Set the total energy of cosmic ray proton.
        
        Parameters
        ----------
        - Ucr (quantity): the energy in unit of GeV/cm3, can be an array matching radius
        - radius (quantity): the radius matching Ucr in case Jp is a 2d function
        - Emin (quantity) : the minimal energy for integration
        - Emax (quantity) : the maximum integration. This is important for 
        slopes <~ 2. From Pinzke et al 2010, we expect a cutoff at E~10^10 GeV.
        
        Outputs
        --------
        """
        
        U0 = self.get_cr_energy_density(radius=radius, Emin=Emin, Emax=Emax)
        rescale = (Ucr/U0).to_value('')
        
        self._norm = rescale
