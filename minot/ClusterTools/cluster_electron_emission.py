"""
This file contains the ClusterElectronEmission class. It aims at computing
the synchrotron and inverse Compton emission for a given electron poulation. 
The synchrotron calculation is based on Aharonian, Kelner and Prosekin 2010 
and inspired by Naima.
"""


#==================================================
# Requested imports
#==================================================

import numpy as np
import astropy.units as u
from astropy import constants as const
from scipy.special import cbrt
from astropy.cosmology import Planck15 as cosmo

#==================================================
# Class
#==================================================

class ClusterElectronEmission(object):
    """ Observable class
    This class compute synchrotron and inverse compton emission for a population 
    of relativistic electrons in a magnetic field in in CMB photon field.
    At high energy, the IC emission analytical parametrization present sharp features 
    which require a rather high NptEePD (10 is clearly to low and will induce wiggles 
    in the spectrum)

    Attributes
    ----------  
    - Je (function): a function which take the electron energy (in GeV) as input
    and returns the number of cosmic ray electron in unit of GeV-1. 
    Je can also depend on radius (in kpc), in which case the radius should be provided 
    when calling the spectrum and Je will be in GeV-1 cm-3. Je is defined 
    as Je(radius, energy)[i_radius, j_energy].
    - Eemin (quantity): minimal electron energy
    - Eemax (quantity): maximum electron energy
    - NptEePD (int): number of electron energy per decade
    - _norm (float): the normalization of the electron spectrum

    Methods
    ----------  
    - synchrotron(self, Egamma_input, Bfield, ): compute the gamma ray spectrum
    - inverse_compton(self, Egamma_input): compute the inverse compton spectrum
    - get_cr_energy_density(self, Emin=None, Emax=None): compute the energy stored in the CR electron
    - set_cr_energy_density(self, Ucr, Emin=None, Emax=None): set the energy stored in the CR electron
    """
    
    #========== Init
    def __init__(self, Je,
                 Eemin=None,
                 Eemax=None,
                 NptEePd=100):

        self._norm = 1.0

        self.Je = Je
        if Eemin is None:
            Eemin = (const.m_e *const.c**2).to_value('GeV')
        else:
            Eemin = Eemin.to_value('GeV')
        if Eemax is None:
            Eemax = 1e7
        else:
            Eemax = Eemax.to_value('GeV')
        self._Eemin = Eemin
        self._Eemax = Eemax
        self._NptEePd = NptEePd
        self.Ee = np.logspace(np.log10(Eemin), np.log10(Eemax), int(NptEePd*(np.log10(Eemax/Eemin))))
        
    
    #========== Synchrotron emission
    def synchrotron(self, Ephoton_input, radius_input=None, B=1.0*u.uG):
        """
        Compute the synchrotron emission
        
        Parameters
        ----------
        - energy (quantity): energy array homogeneous to GeV
        - radius (quantity) : array of radius to be provided if physial
        parameters are position dependant
        - B (quantity): magnetic field strength homogeneous to Gauss
        
        Outputs
        --------
        - Energy loss (GeV/s)
        """

        # Check input photons
        Ephoton = Ephoton_input.to('eV')
        if type(Ephoton) == float: Ephoton = np.array([Ephoton])

        # Check electron energy
        Ee = self.Ee                                        # 1D
        gamma = (Ee/(const.m_e*const.c**2).to_value('GeV')) # 1D

        ampli = np.sqrt(3)/(8*np.pi**2)*(const.e.value*const.e.unit)**3/(const.eps0*const.m_e*const.c*const.hbar)

        #---------- Case of integrated quantities, no radius
        if radius_input is None:
            # Get the electron distribution
            Je = self._norm * self.Je(Ee) * u.GeV**-1  # 1D

            # Get the normalization function (1D: Ephot)
            func = ampli * B / Ephoton
        
            # Get the critical energy and energy ratio
            Ec = (3.0/2)*gamma**2*const.e.value*const.e.unit*B*const.hbar / (const.m_e) # 1D: Eelec
            EphotEc = Ephoton.to_value('GeV') / np.vstack(Ec.to_value('GeV')) # 2D: photon energy, electron energy

            # Compute integrand
            dNphot_dEdt = func * self._Gtilde(EphotEc)
        
            # Integrate over electron energy
            dNphot_dEdt = self._trapz_loglog(np.vstack(Je.value)*Je.unit * dNphot_dEdt, Ee*u.GeV, axis=0)

            # Get the output
            output = dNphot_dEdt.to('GeV-1 s-1')

        #---------- Case of differential quantities: function of radius
        # It might be best trying to compute arrays, avoiding loop, but in 3d...
        else:
            # Check the radius
            radius = radius_input.to('kpc')
            if type(radius) == float: Ephoton = np.array([radius])

            # Get the electron distribution
            if type(self._norm) == float: # Case where CRe normalisation was not set
                norm = self._norm
            else:                         # Case where CRe normalisation was set
                norm = np.vstack(self._norm)
            Je = norm * self.Je(radius.to_value('kpc'), Ee)*u.GeV**-1*u.cm**-3 # 1D
                
            output = np.zeros((len(radius), len(Ephoton))) * u.GeV**-1*u.cm**-3*u.s**-1
            
            for i in range(len(radius)):
                if B[i] != 0: # no need to compute for B==0
                    # Get the Je at given radius
                    Je_i = Je[i, :]
                
                    # Get the normalization function
                    func = ampli * B[i] / Ephoton # 1D: Ephot
            
                    # Get the critical energy and energy ratio
                    Ec = (3.0/2)*gamma**2*const.e.value*const.e.unit*B[i]*const.hbar / (const.m_e) # 1D: Eelec
                    EphotEc = Ephoton.to_value('GeV') / np.vstack(Ec.to_value('GeV')) # 2D: photon energy, electron energy
                    
                    # Compute integrand
                    dNphot_dEdt = func * self._Gtilde(EphotEc) # 3D: radius, photon energy, electron energy
                    
                    # Integrate over electron energy
                    dNphot_dEdVdt = self._trapz_loglog(np.vstack(Je_i.value)*Je_i.unit * dNphot_dEdt, Ee*u.GeV, axis=0) # 1D:Ephot
                    
                    # Get the output
                    output[i,:] = dNphot_dEdVdt.to('GeV-1 cm-3 s-1')
            
        return output

    #========== Inverse Compton loss
    def inverse_compton(self, Egamma_input, radius_input=None, redshift=0.0):
        """
        Compute inverse Compton emission for a Black Body spectrum at Tcmb.
        From Khangulyan, Aharonian, & Kelner 2014, eq 14.
        
        Parameters
        ----------
        - energy (quantity): energy array homogeneous to GeV
        - radius (quantity) : array of radius to be provided if physial
        parameters are position dependant
        - redshift (float): cluster redshift
        
        Outputs
        --------
        - Energy loss (GeV/s)
        """

        #---------- Extract quantities to be used, normalized by electron rest mass
        Ecmb = (const.k_B*cosmo.Tcmb0*(1+redshift) / (const.m_e*const.c**2)).to_value('')       # kB T/m_e c^2

        Egamma = (Egamma_input / (const.m_e*const.c**2)).to_value('')                           # E_photon/m_e c^2
        if type(Egamma) == float: Egamma = np.array([Egamma])
        Egamma = np.vstack(Egamma)
        Eelec = (self.Ee/(const.m_e*const.c**2).to_value('GeV'))                                # E_elec/m_e c^2

        #---------- Conput the normalizing constant
        r0 = (const.e.value*const.e.unit)**2 / (const.m_e*const.c**2) / (4*np.pi*const.eps0)
        C0 = 2 * r0**2 * const.m_e**3 * const.c**4 / (np.pi * const.hbar**3)                    # Should be s-1
        
        #---------- Compute equation the cross section from Eq 14
        par3 = [0.606, 0.443, 1.481, 0.540, 0.319] # Parameters from Eqs 26, 27
        par4 = [0.461, 0.726, 1.457, 0.382, 6.620]

        z = Egamma / Eelec
        z[z>=1] = 0 # to avoid error, which are set to zero anyway afterwards
        x = z/(1-z) / (4.0 * Eelec * Ecmb)
        C1 = (z**2/(2*(1-z))*self._G34(x, par3) + self._G34(x, par4))
        dNph_dEdt0 = C0.to_value('s-1')*(Ecmb/Eelec)**2*C1/(const.m_e*const.c**2).to_value('GeV')
        
        good = (Egamma < Eelec) * (Eelec > 1)
        dNph_dEdt1 = np.where(good, dNph_dEdt0, np.zeros_like(dNph_dEdt0))
        dNph_dEdt = dNph_dEdt1 * u.GeV**-1 * u.s**-1
        
        #---------- Integrate the cross section over the CRe population
        Ee = self.Ee
        if radius_input is None:
            Je = self._norm * self.Je(Ee) * u.GeV**-1           # 1D
            spec = self._trapz_loglog(Je * dNph_dEdt, Ee*u.GeV) # GeV-1 x s-1 x GeV-1 x GeV
            output = spec.to('GeV-1 s-1')
            
        else:
            # Check the radius
            radius = radius_input.to('kpc')
            if type(radius) == float: Ephoton = np.array([radius])
            
            # Get the electron distribution
            if type(self._norm) == float: # Case where CRe normalisation was not set
                norm = self._norm
            else:                         # Case where CRe normalisation was set
                norm = np.vstack(self._norm)
            Je = norm * self.Je(radius.to_value('kpc'), Ee)*u.GeV**-1*u.cm**-3 # 1D
                                
            output = np.zeros((len(radius), len(Egamma))) * u.GeV**-1*u.cm**-3*u.s**-1
            
            for i in range(len(radius)):
                Je_i = Je[i, :]
                spec = self._trapz_loglog(Je_i * dNph_dEdt, Ee*u.GeV) # GeV-1 x cm-3 x s-1 x GeV-1 x GeV
                output[i,:] = spec.to('GeV-1 cm-3 s-1')

        return output


    #========== Get the total energy in CR
    def get_cr_energy_density(self, radius=None, Emin=None, Emax=None):
        """ 
        Compute the total energy of cosmic ray electron
        
        Parameters
        ----------
        - radius (quantity): the radius in case Je is a 2d function
        - Emin (quantity) : the minimal energy for integration
        - Emax (quantity) : the maximum integration. This is important for 
        slopes <~ 2. From Pinzke et al 2010, we expect a cutoff at E~10^10 GeV.
        
        Outputs
        --------    
        - U_CR = the energy in GeV/cm^-3
        """
        
        #----- Def Emin and Emax
        if Emin is None:
            Emin_lim = self._Eemin
        else:
            Emin_lim = Emin.to_value('GeV')
        
        if Emax is None:
            Emax_lim = self._Eemax
        else:
            Emax_lim = Emax.to_value('GeV')

        Ee = np.logspace(np.log10(Emin_lim), np.log10(Emax_lim), int(self._NptEePd*(np.log10(Emax_lim/Emin_lim))))

        #----- Integration        
        if radius is None:
            U_CR = self._norm * self._trapz_loglog(self._JeEe(Ee), Ee) # GeV/cm^-3
        else:
            U_CR = self._norm * self._trapz_loglog(self._JeEe(Ee, radius=radius.to_value('kpc')), Ee, axis=1) # GeV/cm^-3 = f(r)

        return U_CR * u.GeV / u.cm**3
    
    #========== Get the total energy in CR
    def set_cr_energy_density(self, Ucr, radius=None, Emin=None, Emax=None):
        """ 
        Set the total energy of cosmic ray electron.
        
        Parameters
        ----------
        - Ucr (quantity): the energy in unit of GeV/cm3, can be an array matching radius
        - radius (quantity): the radius matching Ucr in case Je is a 2d function
        - Emin (quantity) : the minimal energy for integration
        - Emax (quantity) : the maximum integration. This is important for 
        slopes <~ 2. From Pinzke et al 2010, we expect a cutoff at E~10^10 GeV.
        
        Outputs
        --------
        """
        
        U0 = self.get_cr_energy_density(radius=radius, Emin=Emin, Emax=Emax)
        rescale = (Ucr/U0).to_value('')
        
        self._norm = rescale

    #========== Je*Ep
    def _JeEe(self, Ee, radius=None):
        """ 
        Multiply the electron distribution by the energy
        
        Parameters
        ----------
        - Ee (GeV) : the electron energy
        - radius (kpc): the radius in case Je is a 2d function
        
        Outputs
        --------    
        - Ee Je(Ep)
        """

        if radius is None:
            func = Ee * self.Je(Ee)
        else:
            func = Ee*self.Je(radius, Ee) #Ep and Je have different dim, but this is ok with python
            
        return func
    
    
    #========== Useful equation for synchrotron emission
    def _Gtilde(self, x):
        """
        Useful equation. Aharonian, Kelner, Prosekin 2010 Eq. D7
        Taken from Naima.
        
        Factor ~2 performance gain in using cbrt(x)**n vs x**(n/3.)
        Invoking crbt only once reduced time by ~40%
        """
        cb = cbrt(x) # x**1/3
        gt1 = 1.808 * cb / np.sqrt(1 + 3.4 * cb ** 2.0)
        gt2 = 1 + 2.210 * cb ** 2.0 + 0.347 * cb ** 4.0
        gt3 = 1 + 1.353 * cb ** 2.0 + 0.217 * cb ** 4.0
        
        return gt1 * (gt2 / gt3) * np.exp(-x)

    #========== Useful equation for IC emission
    def _G34(self, x, par):
        """
        Eqs 20, 24, 25 of Khangulyan et al (2014). 
        Taken from Naima.
        """
        
        alpha, a, beta, b, c = par
        
        # Eq 25
        f1 = np.pi ** 2 / 6.0
        f2 = (1 + c * x) / (1 + f1 * c * x)
        G  = f1 * f2 * np.exp(-x)
        
        # Eq 20        
        f3 = 1 + b * x ** beta
        g = 1.0 / (a * x ** alpha / f3 + 1.0)
        
        return G * g

        
    #========== Integration loglog space with trapezoidale rule
    def _trapz_loglog(self, y, x, axis=-1, intervals=False):
        """
        Integrate along the given axis using the composite trapezoidal rule in
        loglog space. Integrate y(x) along given axis in loglog space. y can be a function 
        with multiple dimension. This follows the script in the Naima package.
        
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

