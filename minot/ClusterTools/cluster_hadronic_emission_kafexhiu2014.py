"""
This file contains the PPmodel class. It aims at computing
the gamma, electron and neutrinos spectrum for hadronic interactions between 
cosmic rays and ICM. It is based on Kafexhiu et al. 2014, and compute spectra 
given as dN/dEdVdt in GeV-1 cm-3 s-1.
It is inspired by the Naima package.

"""

#==================================================
# Requested imports
#==================================================

import numpy as np
from astropy import constants as const
import astropy.units as u

from minot.ClusterTools import cluster_hadronic_emission_kelner2006

#==================================================
# Class
#==================================================

class PPmodel(object):
    """ Observable class
    This class compute gamma, electron and eventually neutrinos spectra
    for hadronic interaction p_gas + p_CR.

    Attributes
    ----------  
    - Jp (function): a function which take the proton energy (in GeV) as input
    and returns the number of cosmic ray proton in unit of GeV-1 cm-3. 
    Jp can also depend on radius (in kpc), in which case the radius should be provided 
    when calling the spectrum. Jp is defined as Jp(radius, energy)[i_radius, j_energy].

    - abundance (float): in unit of solar metallicity (0.3 is typical for cluster). 
    This assumes a rescaling in the abundance measured by
    - Yp (float): helium mass fraction (BBN give 0.245 and is default value)
    - hiEmodel (str): 'Pythia8', 'SIBYLL', 'QGSJET', 'Geant4'
    - Epmin (quantity): minimal proton energy
    - Epmax (quantity): maximum proton energy
    - NptEpPD (int): number of proton energy per decade

    Methods
    ----------  
    - gamma_spectrum(self, Egamma_input, limit='mixed'): compute the gamma ray spectrum
    - get_cr_energy_density(self, Emin=None, Emax=None): compute the energy stored in the CR protons
    - set_cr_energy_density(self, Ucr, Emin=None, Emax=None): set the energy stored in the CR protons

    """
    
    #========== Init
    def __init__(self,
                 Jp,
                 Y0=0.2735,
                 Z0=0.0153,
                 abundance=0.3,
                 hiEmodel="Pythia8",
                 Epmin=None,
                 Epmax=None,
                 NptEpPd=100
    ):

        #----- Base 'hidden' parameters
        self._m_pi = 0.1349766
        self._m_p = (const.m_p*const.c**2).to_value('GeV')
        self._Tth =  2*self._m_pi + self._m_pi**2 / 2.0 / self._m_p
        self._norm = 1.0

        # Abundance of heavy elements relative to H in number.
        # The ration between elements is the same as the solar one (Mernier et al 2019, Arxiv:1811.01967)
        # Only the most common in mass are used (Lodders 2009, Table 10)
        He_frac = Y0 / 4.0 / (1.0-Y0-Z0)
        Ncr = ['H', 'He',    'C',     'N',    'O',    'Ne',   'Mg',    'Si',    'S',     'Fe'] # Target ICM elements
        Acr = [1,   4,       12,      14,     16,     20,     24,      28,      32,      56]
        Ycr = [1,   He_frac, 2.78e-04*abundance, 8.19e-05*abundance, 6.06e-04*abundance,
               1.27e-04*abundance, 3.98e-05*abundance, 3.86e-05*abundance, 1.63e-05*abundance, 3.27e-05*abundance]
        Nta = ['H', 'He',    'C',     'N',    'O',    'Ne',   'Mg',    'Si',    'S',     'Fe']
        Ata = [1,   4,       12,      14,     16,     20,     24,      28,      32,      56]
        Yta = [1,   He_frac, 2.78e-04*abundance, 8.19e-05*abundance, 6.06e-04*abundance,
               1.27e-04*abundance, 3.98e-05*abundance, 3.86e-05*abundance, 1.63e-05*abundance, 3.27e-05*abundance]
        abund = {}
        abund['Name_p'] = Ncr
        abund['A_p']    = Acr
        abund['Y_p']    = Ycr
        abund['Name_t'] = Nta
        abund['A_t']    = Ata
        abund['Y_t']    = Yta
        self._abund = abund
        if Y0 == 0 and abundance == 0 :
            self._nuclear_enhancement = False
        else:
            self._nuclear_enhancement = True
                
        # Table VII
        b = {}
        b['Geant4_0'] = [9.530, 0.5200, 0.054]   # 1 <= Tp < 5
        b['Geant4']   = [9.130, 0.3500, 9.7e-3]  # Tp >= 5
        b['Pythia8']  = [9.060, 0.3795, 0.01105] # Tp >  50
        b['SIBYLL']   = [10.77, 0.4120, 0.01264] # Tp >  100
        b['QGSJET']   = [13.16, 0.4419, 0.01439] # Tp >  100
        self._b = b
        
        # Table IV
        a = {}
        a['Geant4']  = [0.728, 0.5960, 0.491, 0.2503, 0.117]  # Tp > 5
        a['Pythia8'] = [0.652, 0.0016, 0.488, 0.1928, 0.483]  # Tp > 50
        a['SIBYLL']  = [5.436, 0.2540, 0.072, 0.0750, 0.166]  # Tp > 100
        a['QGSJET']  = [0.908, 0.0009, 6.089, 0.1760, 0.448]  # Tp > 100
        self._a = a

        # table V data (np.nan indicate that functions of Tp are needed and are defined as need in function F)
        # parameter order is lambda, alpha, beta, gamma
        F_mp = {}
        F_mp['ExpData']  = [1.00, 1.0, np.nan, 0.0]     # Tth  <= Tp <= 1.0
        F_mp['Geant4_0'] = [3.00, 1.0, np.nan, np.nan]  # 1.0  <  Tp <= 4.0
        F_mp['Geant4_1'] = [3.00, 1.0, np.nan, np.nan]  # 4.0  <  Tp <= 20.0
        F_mp['Geant4_2'] = [3.00, 0.5, 4.2,    1.0]     # 20.0 <  Tp <= 100
        F_mp['Geant4']   = [3.00, 0.5, 4.9,    1.0]     # Tp > 100
        F_mp['Pythia8']  = [3.50, 0.5, 4.0,    1.0]     # Tp > 50
        F_mp['SIBYLL']   = [3.55, 0.5, 3.6,    1.0]     # Tp > 100
        F_mp['QGSJET']   = [3.55, 0.5, 4.5,    1.0]     # Tp > 100
        self._F_mp = F_mp

        # Energy at which each of the hiE models start being valid
        self._Etrans = {"Pythia8": 50, "SIBYLL": 100, "QGSJET": 100, "Geant4": 100}

        #----- Deal with input parameters
        if Epmin is None:
            Epmin = (self._m_p + 2*self._m_pi + self._m_pi**2 / 2.0 / self._m_p)*(1+1e-10)
        else:
            if Epmin.to_value('GeV') > self._Tth+self._m_p:
                Epmin = Epmin.to_value('GeV')
            else:
                Epmin = (self._m_p + 2*self._m_pi + self._m_pi**2 / 2.0 / self._m_p)*(1+1e-10)
                
        if Epmax is None:
            Epmax = 1e7
        else:
            Epmax = Epmax.to_value('GeV')

        self._Epmin = Epmin
        self._Epmax = Epmax
        self._NptEpPd = NptEpPd

        #----- Include input parameters
        self.Jp = Jp
        self.Ep = np.logspace(np.log10(Epmin), np.log10(Epmax), int(NptEpPd*(np.log10(Epmax/Epmin))))
        self.hiEmodel = hiEmodel

    #========== Compute the spectrum
    def gamma_spectrum(self, Egamma_input, radius_input=None, nH=1.0*u.cm**-3):
        """
        Compute the gamma ray spectrum.

        Parameters
        ----------
        - Egamma_input (quantity) : the gamma ray energy vector
        - radius_input (quantity) : array of radius to be  provided if Jp is 2D
        - nH (quantity) : in case radius is provided, nH should match the radius

        Outputs
        --------    
        - The photon spectrum in unit homogeneous to GeV-1 cm-3 s-1
        """
        
        Egamma = Egamma_input.to_value('GeV')
        Ep     = self.Ep

        #---------- Case of radius provided
        if radius_input is None:
            Jp = self.Jp(Ep)
            norm = const.c.to_value('cm/s') * nH.to_value('cm-3') * self._norm

            spec = np.zeros(len(Egamma))
            for i in range(len(Egamma)):
                diffsigma = self._diffsigma(Ep, Egamma[i])
                spec[i] = norm*self._trapz_loglog(diffsigma * Jp, Ep)

        #---------- Case of 1D, without radius
        else:
            if len(nH) != len(radius_input):
                raise ValueError('nH should have the same size as radius_input')
            
            radius = radius_input.to_value('kpc')
            Jp = self.Jp(radius, Ep)
            
            spec0 = np.zeros((len(radius), len(Egamma)))
            for i in range(len(Egamma)):
                diffsigma = self._diffsigma(Ep, Egamma[i])
                spec0[:,i] = self._trapz_loglog(diffsigma * Jp, Ep, axis=1)
                
            norm = const.c.to_value('cm/s') * nH.to_value('cm-3') * self._norm
            norm_grid = (np.tile(norm, [len(Egamma),1])).T
            spec = spec0 * norm_grid
            
        return spec * u.GeV**(-1) * u.cm**(-3) * u.s**(-1)

    #========== Compute the electron spectrum
    def electron_spectrum(self, Ee_input, radius_input=None, nH=1.0*u.cm**-3):
        """
        Compute the electron spectrum.

        Parameters
        ----------
        - Ee_input (quantity) : the electron energy vector in unit of GeV
        - radius_input (quantity) : array of radius to be  provided if Jp is 2D
        - nH (quantity) : in case radius is provided, nH should match the radius
        
        Outputs
        --------    
        - The electron spectrum in unit homogeneous to GeV-1 cm-3 s-1
        """

        spec_gamma = self.gamma_spectrum(Ee_input, radius_input=radius_input, nH=nH) # Compute the gamma ray spectrum

        # Use Kelner2016 and assume the ratio remain the same
        K06 = cluster_hadronic_emission_kelner2006.PPmodel(self.Jp,
                                                           Epmin=self._Epmin*u.GeV, Epmax=self._Epmax*u.GeV,
                                                           NptEpPd=self._NptEpPd)
        
        spec_gamma_kelner = K06.gamma_spectrum(Ee_input, radius_input=radius_input, nH=nH)
        spec_elec_kelner = K06.electron_spectrum(Ee_input, radius_input=radius_input, nH=nH)

        wbad = (spec_gamma_kelner <= 0)
        spec_gamma_kelner[wbad] = 1.0*spec_gamma_kelner.unit
        
        spec = spec_elec_kelner * (spec_gamma/spec_gamma_kelner).to_value('')
        spec[wbad] *= 0

        return spec

    #========== Compute the electron spectrum
    def neutrino_spectrum(self, Enu_input, radius_input=None, nH=1.0*u.cm**-3, flavor='numu'):
        """
        Compute the neutrino spectrum.

        Parameters
        ----------
        - Enu_input (quantity) : the neutrino energy vector in unit of GeV
        - radius_input (quantity) : array of radius to be  provided if Jp is 2D
        - nH (quantity) : in case radius is provided, nH should match the radius

        Outputs
        --------    
        - The neutrino spectrum in unit homogeneous to GeV-1 cm-3 s-1
        """

        spec_gamma = self.gamma_spectrum(Enu_input, radius_input=radius_input, nH=nH) # Compute the gamma ray spectrum

        # Use Kelner2016 and assume the ratio remain the same
        K06 = cluster_hadronic_emission_kelner2006.PPmodel(self.Jp,
                                                           Epmin=self._Epmin*u.GeV, Epmax=self._Epmax*u.GeV,
                                                           NptEpPd=self._NptEpPd)
        
        spec_gamma_kelner = K06.gamma_spectrum(Enu_input, radius_input=radius_input, nH=nH)
        spec_nu_kelner = K06.neutrino_spectrum(Enu_input, radius_input=radius_input, nH=nH, flavor=flavor)

        wbad = (spec_gamma_kelner <= 0)
        spec_gamma_kelner[wbad] = 1.0*spec_gamma_kelner.unit
        
        spec = spec_nu_kelner * (spec_gamma/spec_gamma_kelner).to_value('')
        spec[wbad] *= 0
                
        return spec
    
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

    #========== Differential cross section
    def _diffsigma(self, Ep, Egamma):
        """
        Compute the differential cross section
        dsigma/dEgamma = Amax(Tp) F(Tp, Egamma) as eq 8 of Kafexhiu2014

        Parameters
        ----------
        - Ep (GeV) : the proton energy
        - Egamma (GeV) : the gamma ray energy
        
        Outputs
        --------    
        - dsigma/dEgamma
        """
        
        Tp = Ep - self._m_p

        diffsigma = self._Amax(Tp) * self._F(Tp, Egamma)

        if self._nuclear_enhancement:
            diffsigma *= self._nuclear_factor(Tp)

        return diffsigma

    #========== Amplitude max
    def _Amax(self, Tp):
        """
        Compute the Amax term from Eq 12, which give the
        amplitude of the spectrum.

        Parameters
        ----------
        - Tp (GeV) : the proton kinetic energy
        
        Outputs
        --------    
        - Amax: the amplitude
        """
        
        loE = np.where(Tp < 1.0)
        hiE = np.where(Tp >= 1.0)
        Amax = np.zeros(Tp.size)
        b    = self._b_params(Tp)
        
        EpimaxLAB = self._calc_EpimaxLAB(Tp)
        Amax[loE] = b[0] * self._sigma_pi(Tp[loE]) / EpimaxLAB[loE]
        
        thetap = Tp / self._m_p
        Amax[hiE] = (b[1] * thetap[hiE] ** -b[2]
                     * np.exp(b[3] * np.log(thetap[hiE]) ** 2)
                     * self._sigma_pi(Tp[hiE]) / self._m_p)
        
        return Amax

    #========== Get the b parameters
    def _b_params(self, Tp):
        """
        Return the b parameters of Eq 12

        Parameters
        ----------
        - Tp (GeV) : the proton kinetic energy
        
        Outputs
        --------    
        - b: numerical parameters in tuple
        """
        
        b0 = 5.9
        
        hiE = np.where(Tp >= 1.0)
        TphiE = Tp[hiE]
        b1 = np.zeros(TphiE.size)
        b2 = np.zeros(TphiE.size)
        b3 = np.zeros(TphiE.size)
        
        idx = np.where(TphiE < 5.0)
        b1[idx], b2[idx], b3[idx] = self._b["Geant4_0"]
        
        idx = np.where(TphiE >= 5.0)
        b1[idx], b2[idx], b3[idx] = self._b["Geant4"]
        
        idx = np.where(TphiE >= self._Etrans[self.hiEmodel])
        b1[idx], b2[idx], b3[idx] = self._b[self.hiEmodel]
        
        return b0, b1, b2, b3

    #========== EpimaxLAB
    def _calc_EpimaxLAB(self, Tp):
        """
        Return the parameters of Eq 10

        Parameters
        ----------
        - Tp (GeV) : the proton kinetic energy
        
        Outputs
        --------    
        - Epi_max_LAB
        """
        
        m_p = self._m_p
        m_pi = self._m_pi
        
        s = 2 * m_p * (Tp + 2 * m_p)  # center of mass energy
        EpiCM = (s - 4 * m_p ** 2 + m_pi ** 2) / (2 * np.sqrt(s))
        PpiCM = np.sqrt(EpiCM ** 2 - m_pi ** 2)
        gCM = (Tp + 2 * m_p) / np.sqrt(s)
        betaCM = np.sqrt(1 - gCM ** -2)
        EpimaxLAB = gCM * (EpiCM + PpiCM * betaCM)
        
        return EpimaxLAB

    #========== sigma pi and subfunctions
    def _sigma_pi(self, Tp):
        """
        Return the parameters sigma_pi used in Eq 12
        and def in section 2.4

        Parameters
        ----------
        - Tp (GeV) : the proton kinetic energy
        
        Outputs
        --------    
        - sigma_pi
        """
        
        sigma = np.zeros_like(Tp)

        # for E<2GeV (Fit from experimental data section 2.B)
        idx1 = np.where(Tp < 2.0)
        sigma[idx1] = self._sigma_pi_loE(Tp[idx1])
        
        # for 2GeV<=E<5GeV (Eq6 & 1: Geant 4.10.0 model)
        idx2 = np.where((Tp >= 2.0) * (Tp < 5.0))
        sigma[idx2] = self._sigma_pi_midE(Tp[idx2])
        
        # for 5GeV<=E<Etrans (Eq 7 & 1)
        idx3 = np.where((Tp >= 5.0) * (Tp < self._Etrans[self.hiEmodel]))
        sigma[idx3] = self._sigma_pi_hiE(Tp[idx3], self._a["Geant4"]) # Use Geant4 at intermediate energies
        
        # for E>=Etrans (Eq 7 & 1)
        idx4 = np.where((Tp >= self._Etrans[self.hiEmodel]))
        sigma[idx4] = self._sigma_pi_hiE(Tp[idx4], self._a[self.hiEmodel])

        return sigma

    def _sigma_pi_hiE(self, Tp, a):
        m_p = self._m_p
        csip = (Tp - 3.0) / m_p
        m1 = a[0] * csip ** a[3] * (1 + np.exp(-a[1] * csip ** a[4]))
        m2 = 1 - np.exp(-a[2] * csip ** 0.25)
        multip = m1 * m2
        return self._sigma_inel(Tp) * multip
    
    def _sigma_pi_midE(self, Tp):
        m_p = self._m_p
        Qp = (Tp - self._Tth) / m_p
        multip = -6e-3 + 0.237 * Qp - 0.023 * Qp ** 2
        return self._sigma_inel(Tp) * multip

    def _sigma_pi_loE(self, Tp):
        m_p = self._m_p
        m_pi = self._m_pi

        #----- one pion production
        Mres = 1.1883  # GeV
        Gres = 0.2264  # GeV
        s = 2 * m_p * (Tp + 2 * m_p)  # center of mass energy
        gamma = np.sqrt(Mres ** 2 * (Mres ** 2 + Gres ** 2)) # Eq 4
        K = (np.sqrt(8) * Mres * Gres * gamma) / (np.pi * np.sqrt(Mres ** 2 + gamma)) # Eq 4
        fBW = m_p * K / (((np.sqrt(s) - m_p) ** 2 - Mres ** 2) ** 2 + Mres ** 2 * Gres ** 2) #Eq 4
        mu = np.sqrt((s - m_pi ** 2 - 4 * m_p ** 2) ** 2 - 16 * m_pi ** 2 * m_p ** 2) / (2 * m_pi * np.sqrt(s)) # Eq 3
        sigma0 = 7.66e-3  # mb
        sigma1pi = sigma0 * mu ** 1.95 * (1 + mu + mu ** 5) * fBW ** 1.86 # Eq 2 (mb)

        #----- two pion production
        sigma2pi = 5.7 / (1 + np.exp(-9.3 * (Tp - 1.4))) #Eq 5 (mb)
        E2pith = 0.56  # GeV
        sigma2pi[np.where(Tp < E2pith)] = 0.0

        return (sigma1pi + sigma2pi) * 1e-27  # return in cm-2

    
    def _sigma_inel(self, Tp):
        """
        Inelastic cross-section for p-p interaction (Eq 1)

        Parameters
        ----------
        - Tp (GeV) : the proton kinetic energy

        Outputs
        -------
        sigma_inel (float, cm2)
        """
        L = np.log(Tp / self._Tth)
        sigma = 30.7 - 0.96 * L + 0.18 * L ** 2
        sigma *= (1 - (self._Tth / Tp) ** 1.9) ** 3
        return sigma * 1e-27  # convert from mbarn to cm-2


    #========== The function F
    def _F(self, Tp, Egamma):
        """
        Run the computing of the function F from eq 8, which describe the 
        shape of the spectrum. The function is given in eq 11. The model 
        parameters depend on energy.

        Parameters
        ----------
        - Tp (GeV) : the proton kinetic energy
        - Egamma (GeV) : the gamma energy

        Outputs
        -------
        - F
        """
        
        F = np.zeros_like(Tp)
        # below Tth
        F[np.where(Tp < self._Tth)] = 0.0

        #----- Tth <= E <= 1GeV: Experimental data
        idx = np.where((Tp >= self._Tth) * (Tp <= 1.0))
        if idx[0].size > 0:
            kappa = self._kappa(Tp[idx])                                        
            mpar = self._F_mp["ExpData"]
            mpar[2] = kappa
            F[idx] = self._F_func(Tp[idx], Egamma, mpar)

        #----- 1GeV < Tp < 4 GeV: Geant4 model 0
        idx = np.where((Tp > 1.0) * (Tp <= 4.0))
        if idx[0].size > 0:
            mpar = self._F_mp["Geant4_0"]
            mu = self._mu(Tp[idx])                                              
            mpar[2] = mu + 2.45
            mpar[3] = mu + 1.45
            F[idx] = self._F_func(Tp[idx], Egamma, mpar)

        #----- 4 GeV < Tp < 20 GeV
        idx = np.where((Tp > 4.0) * (Tp <= 20.0))
        if idx[0].size > 0:
            mpar = self._F_mp["Geant4_1"]
            mu = self._mu(Tp[idx])
            mpar[2] = 1.5 * mu + 4.95
            mpar[3] = mu + 1.50
            F[idx] = self._F_func(Tp[idx], Egamma, mpar)

        #----- 20 GeV < Tp < 100 GeV
        idx = np.where((Tp > 20.0) * (Tp <= 100.0))
        if idx[0].size > 0:
            mpar = self._F_mp["Geant4_2"]
            F[idx] = self._F_func(Tp[idx], Egamma, mpar)

        #----- Tp > Etrans
        idx = np.where(Tp > self._Etrans[self.hiEmodel])
        if idx[0].size > 0:
            mpar = self._F_mp[self.hiEmodel]
            F[idx] = self._F_func(Tp[idx], Egamma, mpar)

        return F

    #========= F function model
    def _F_func(self, Tp, Egamma, modelparams):
        """
        Compute the function F from eq 22.

        Parameters
        ----------
        - Tp (GeV) : the proton kinetic energy
        - Egamma (GeV) : the gamma energy
        - modelparams: sub-function parameters which depend on energy range

        Outputs
        -------
        - F
        """
        lamb, alpha, beta, gamma = modelparams
        m_pi = self._m_pi

        #----- Eq 9
        Egmax = self._calc_Egmax(Tp)
        Yg = Egamma + m_pi ** 2 / (4 * Egamma)
        Ygmax = Egmax + m_pi ** 2 / (4 * Egmax)
        Xg = (Yg - m_pi) / (Ygmax - m_pi)
        
        #----- zero out invalid fields (Egamma > Egmax -> Xg > 1)
        Xg[np.where(Xg > 1)] = 1.0

        #----- Eq 11
        C = lamb * m_pi / Ygmax
        F = ((1 - Xg ** alpha) ** beta) / ((1 + Xg / C) ** gamma)
        
        return F

    #========= Compute Egamma max
    def _calc_Egmax(self, Tp):
        """
        Compute the Egamma max given in Eq 10

        Parameters
        ----------
        - Tp (GeV) : the proton kinetic energy

        Outputs
        -------
        - Egmax
        """
        m_pi = self._m_pi
        EpimaxLAB = self._calc_EpimaxLAB(Tp)
        gpiLAB = EpimaxLAB / m_pi
        betapiLAB = np.sqrt(1 - gpiLAB ** -2)
        Egmax = (m_pi / 2) * gpiLAB * (1 + betapiLAB)

        return Egmax
    
    #========== kappa function
    def _kappa(self, Tp):
        """
        Compute the kappa function (eq 14)

        Parameters
        ----------
        - Tp (GeV) : the proton kinetic energy

        Outputs
        -------
        - kappa
        """
        thetap = Tp / self._m_p
        return 3.29 - thetap ** -1.5 / 5.0

    #========== mu function
    def _mu(self, Tp):
        """
        Compute the mu function (eq 15)

        Parameters
        ----------
        - Tp (GeV) : the proton kinetic energy

        Outputs
        -------
        - mu
        """
        q = (Tp - 1.0) / self._m_p
        x = 5.0 / 4.0
        return x * q ** x * np.exp(-x * q)

    #========== Nuclear enhencement factor
    def _nuclear_factor(self, Tp):
        """
        Compute nuclear enhancement factor

        Parameters
        ----------
        - Tp (GeV) : the proton kinetic energy

        Outputs
        -------
        - epstotal : term to be multiplied to sigma
        """
        
        sigmaRpp = 10 * np.pi * 1e-27
        sigmainel = self._sigma_inel(Tp)
        sigmainel0 = self._sigma_inel(1e3)  # at 1e3 GeV
        f = sigmainel / sigmainel0
        f2 = np.where(f > 1, f, 1.0)
        G = 1.0 + np.log(f2)
        
        # epsilon factors computed from Eqs 21 to 23 with local ISM abundances
        epsC = 1.37
        eps1 = 0.29
        eps2 = 0.1
        epsC, eps1, eps2 = self._epsilon_coeff()


        sigmainel[Tp <= self._Tth] = 1.0 # just to avoid dividing by zero
        epstotal = np.where(Tp > self._Tth, epsC + (eps1 + eps2) * sigmaRpp * G / sigmainel, 0.0)
        
        #----- Fix nuclear enhancement factor as it diverges towards Tp = Tth, fix Tp<1 to eps(1.0) = 1.9141
        if np.any(Tp < 1.0):
            loE = np.where((Tp > self._Tth) * (Tp < 1.0))
            epstotal[loE] = 1.9141

        return epstotal
    
    #========== Nuclear enhencement factor
    def _epsilon_coeff(self):
        """
        Compute the nuclear coefficient given by Eq 21, 22, 23
        
        Parameters
        ----------
        
        Outputs
        -------
        - epsC (float): eq 21
        - eps1 (float): eq 22
        - eps2 (float): eq 23
        """

        sigmaRpp = 10*np.pi*1e-27

        Np = np.array(self._abund['Name_p'])
        Ap = np.array(self._abund['A_p'])
        Yp = np.array(self._abund['Y_p'])

        Nt = np.array(self._abund['Name_t'])
        At = np.array(self._abund['A_t'])
        Yt = np.array(self._abund['Y_t'])

        sigma_R_p_Ap = self._sigma_R(1, Ap)
        sigma_R_p_At = self._sigma_R(At, 1)

        wp = Np != 'H'
        wt = Nt != 'H'

        epsC = 1 + 0.5*np.sum(Ap[wp]*Yp[wp]) + 0.5*np.sum(At[wt]*Yt[wt])
        eps1 = 0.5*np.sum(Yp[wp]*sigma_R_p_Ap[wp]/sigmaRpp) + 0.5*np.sum(Yt[wt]*sigma_R_p_At[wt]/sigmaRpp)
        eps2 = sum(Yp[wp][i]*Yt[wt][j]*(Ap[wp][i]*sigma_R_p_At[wt][j]+At[wt][j]*sigma_R_p_Ap[wp][i])/(2.0*sigmaRpp) 
                   for i in range(np.sum(wp)) for j in range(np.sum(wt)))

        return epsC, eps1, eps2

    #========== Cross section for nuclei interaction
    def _sigma_R(self, Ap, At):
        """
        Compute the nucleus - nucleus cross section
        
        Parameters
        ----------
        - Ap (float or array-like): mass number of cosmic ray element
        - At (float or array-like): mass number of target element
        Only one of the two input can be an array.
        
        Outputs
        -------
        - sigmaR (float): cross section (cm-2)
        """
        beta0 = np.where(Ap == 1, 2.247 - 0.915*(1 + At**(-1.0/3.0)), 1.581 - 0.876*(Ap**(-1.0/3.0) + At**(-1.0/3.0)) )
        #if Ap == 1:
        #    beta0 = 2.247 - 0.915*(1 + At**(-1.0/3.0))
        #else:
        #    beta0 = 1.581 - 0.876*(Ap**(-1.0/3.0) + At**(-1.0/3.0))    
        sigmaR0 = 10*np.pi*1.36**2*1e-27
        sigmaR = sigmaR0 * (Ap**(1.0/3) + At**(1.0/3) - beta0*(Ap**(-1.0/3.0) + At**(-1.0/3.0)))
        
        return sigmaR

