"""
This file contain a subclass of the model.py module and Cluster class. It
is dedicated to the computing of the physical properties of clusters.

"""

#==================================================
# Requested imports
#==================================================

import numpy as np
import copy
from scipy.optimize import brentq
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import os
import astropy.units as u
from astropy import constants as const

from minot              import model_tools
from minot.ClusterTools import cluster_global
from minot.ClusterTools import cluster_profile
from minot.ClusterTools import cluster_spectra
from minot.ClusterTools import cluster_szspec
from minot.ClusterTools import cluster_xspec
from minot.ClusterTools import cluster_hadronic_emission_kafexhiu2014 as K14
from minot.ClusterTools import cluster_hadronic_emission_kelner2006 as K06
from minot.ClusterTools import cluster_electron_loss
from minot.ClusterTools import cluster_electron_emission
    

#==================================================
# Physics class
#==================================================

class Physics(object):
    """ 
    Physics class
    This class searves as a parser to the main Cluster class, to 
    include the subclass Physics in this other file.

    Attributes
    ----------  
    The attributes are the same as the Cluster class, see model.py

    Methods
    ----------  

    * Methods related to the physical description of the cluster

    - get_pressure_gas_profile(self, radius=np.logspace(1,5,1000)*u.kpc): compute the electron
    gas pressure profile.
    - get_density_gas_profile(self, radius=np.logspace(0,4,1000)*u.kpc): compute the electron
    gas density profile.
    - get_temperature_gas_profile(self, radius=np.logspace(0,4,1000)*u.kpc): compute the gas
    temperature profile (same as electron or ions)
    - get_entropy_gas_profile(self, radius=np.logspace(0,4,1000)*u.kpc) : compute the entropy profile
    - get_hse_mass_profile(self, radius=np.logspace(0,4,1000)*u.kpc) : compute the hydrostatic 
    mass profile
    - get_overdensity_contrast_profile(self, radius=np.logspace(0,4,1000)*u.kpc): compute the 
    overdensity contrast.
    - get_mdelta_from_profile(self, delta=500, Rmin=10*u.kpc, Rmax=1e4*u.kpc): compute the mass 
    and radius corresponding to a given overdensity contrast, e.g. M500 or R500, from the hydrostatic
    mass profile and a mean hydrostatic mass bias.
    - get_gas_mass_profile(self, radius=np.logspace(0,4,1000)*u.kpc): compute the gas mass profile
    - get_fgas_profile(self, radius=np.logspace(0,4,1000)*u.kpc): compute the gas fraction profile
    - get_thermal_energy_profile(self, radius=np.logspace(0,4,1000)*u.kpc): compute the thermal
    energy profile
    - get_magfield_profile(self, radius=np.logspace(0,4,1000)*u.kpc): get the magnetic field profile

    - get_normed_density_crp_profile(self, radius=np.logspace(0,4,1000)*u.kpc): get the radial 
    part of the cosmic ray protons distribution, f(r), in dN/dEdV = A f(r) f(E)
    - get_normed_crp_spectrum(self, energy=np.logspace(-2,7,1000)*u.GeV): get the spectral part
    of the cosmic ray proton distribution, f(E), in dN/dEdV = A f(r) f(E)
    - _get_crp_normalization(self): compute the normalization of the cosmic ray proton distribution, 
    A, in dN/dEdV = A f(r) f(E)
    - get_crp_2d(self, energy=np.logspace(-2,7,100)*u.GeV, radius=np.logspace(0,4,100)*u.kpc): compute
    the CRp distribution on a 2d grid (energy vs radius) as dN/dEdV
    - get_density_crp_profile(self, radius=np.logspace(0,4,1000)*u.kpc, Emin=None, Emax=None, 
    Energy_density=False): compute the cosmic ray proton density profile integrating over the energy 
    between Emin and Emax.
    - get_crp_spectrum(self, energy=np.logspace(-2,7,1000)*u.GeV, Rmax=None): compute the cosmic ray proton
    spectrum integrating over the volume up to Rmax
    - get_crp_to_thermal_energy_profile(self, radius=np.logspace(0,4,1000)*u.kpc, Emin=None, Emax=None):
    compute the cosmic ray proton energy (between Emin and Emax) to thermal energy profile.
    
    - get_normed_density_cre1_profile(self, radius=np.logspace(0,4,1000)*u.kpc): get the radial 
    part of the cosmic ray primary electrons distribution, f(r), in dN/dEdV = A f(r) f(E)
    - get_normed_cre1_spectrum(self, energy=np.logspace(-2,7,1000)*u.GeV): get the spectral part
    of the cosmic ray primary electron distribution, f(E), in dN/dEdV = A f(r) f(E)
    - _get_cre1_normalization(self): compute the normalization of the cosmic ray primary electron distribution, 
    A, in dN/dEdV = A f(r) f(E)
    - get_cre1_2d(self, energy=np.logspace(-2,7,100)*u.GeV, radius=np.logspace(0,4,100)*u.kpc): compute
    the CRe1 distribution on a 2d grid (energy vs radius) as dN/dEdV
    - get_density_cre1_profile(self, radius=np.logspace(0,4,1000)*u.kpc, Emin=None, Emax=None, 
    Energy_density=False): compute the cosmic ray primary electron density profile integrating over the energy 
    between Emin and Emax.
    - get_cre1_spectrum(self, energy=np.logspace(-2,7,1000)*u.GeV, Rmax=None): compute the cosmic 
    ray primary electron
    spectrum integrating over the volume up to Rmax
    - get_cre1_to_thermal_energy_profile(self, radius=np.logspace(0,4,1000)*u.kpc, Emin=None, Emax=None):
    compute the cosmic ray primary electron energy (between Emin and Emax) to thermal energy profile.
    
    * Methods related to the physical processes in the cluster

    - get_rate_gamma(self, energy=np.logspace(-2,7,100)*u.GeV, radius=np.logspace(0,4,100)*u.kpc):
    compute the gamma ray emission rate dN/dEdVdt versus energy and radius
    - get_rate_cre2(self, energy=np.logspace(-2,7,100)*u.GeV, radius=np.logspace(0,4,100)*u.kpc):
    compute the CRe production rate dN/dEdVdt versus energy and radius
    - get_rate_neutrino(self, energy=np.logspace(-2,7,100)*u.GeV, radius=np.logspace(0,4,100)*u.kpc, flavor='all'):
    compute the neutrino production rate dN/dEdVdt versus energy and radius
    
    - get_cre2_2d(self, energy=np.logspace(-2,7,100)*u.GeV, radius=np.logspace(0,4,100)*u.kpc):
    compute the CRe population assuming equilibrium dN/dEdV versus energy and radius
    - get_density_cre2_profile(self, radius=np.logspace(0,4,100)*u.kpc,Emin=None, Emax=None, Energy_density=False):
    compute the cosmic ray electron density profile integrating over the energy between Emin and Emax.
    - get_cre2_spectrum(self, energy=np.logspace(-2,7,100)*u.GeV, Rmax=None): compute the cosmic ray electron
    spectrum integrating over the volume up to Rmax
    - get_rate_synchrotron(self, frequency=np.logspace(-3,3,100)*u.GHz, radius=np.logspace(0,4,100)*u.kpc):
    compute the synchrotron emission per unit volume given the electron population and magnetic field
    - get_rate_ic(self, energy=np.logspace(-2,7,100)*u.GeV, radius=np.logspace(0,4,100)*u.kpc):
    compute the inverse compton emission per unit volume given the electron population and CMB(z)

    - get_rate_sz(self, frequency=np.logspace(1,3,100)*u.GHz, radius=np.logspace(0,4,100)*u.kpc, 
    Compton_only=False):
    Compute the SZ emission per cm3

    - make_xspec_table(self, Emin=0.1*u.keV, Emax=2.4*u.keV,Tmin=0.1*u.keV, Tmax=50.0*u.keV, nbin=100,
    nH=0.0/u.cm**2, file_HI=None): compute a temperature versus counts/Sx table to be interpolated
    when getting profiles and maps
    - _itpl_xspec_table(self, xspecfile, Tinput): interpolate xspec tables to chosen temperature
    - get_rate_xray(self, radius=np.logspace(0,4,100)*u.kpc, output_type='C'): compute
    the X-ray emission per unit volume

    """

    
    #==================================================
    # Get the gas electron pressure profile
    #==================================================

    def get_pressure_gas_profile(self, radius=np.logspace(0,4,100)*u.kpc):
        """
        Get the thermal electron pressure profile.
        
        Parameters
        ----------
        - radius (quantity) : the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - p_r (quantity): the electron pressure profile in unit of keV cm-3

        """

        # In case the input is not an array
        radius = model_tools.check_qarray(radius, unit='kpc')

        # get profile
        p_r = self._get_generic_profile(radius, self._pressure_gas_model)
        p_r[radius > self._R_truncation] *= 0

        return radius, p_r.to('keV cm-3')
            
            
    #==================================================
    # Get the gas electron density profile
    #==================================================
    
    def get_density_gas_profile(self, radius=np.logspace(0,4,100)*u.kpc):
        """
        Get the thermal electron density profile.
        
        Parameters
        ----------
        - radius (quantity) : the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - n_r (quantity): the electron density profile in unit of cm-3

        """

        # In case the input is not an array
        radius = model_tools.check_qarray(radius, unit='kpc')

        # get profile
        n_r = self._get_generic_profile(radius, self._density_gas_model)
        n_r[radius > self._R_truncation] *= 0
        
        return radius, n_r.to('cm-3')
    
            
    #==================================================
    # Get the gas electron temperature profile
    #==================================================
    
    def get_temperature_gas_profile(self, radius=np.logspace(0,4,100)*u.kpc):
        """
        Get the thermal temperature profile.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - T_r (quantity): the temperature profile in unit of keV

        """
        
        # In case the input is not an array
        radius = model_tools.check_qarray(radius, unit='kpc')

        # Compute n and P
        radius, n_r = self.get_density_gas_profile(radius=radius)
        radius, P_r = self.get_pressure_gas_profile(radius=radius)

        # Get Temperature
        n_r[n_r <= 0] = np.nan
        T_r = P_r / n_r

        # Apply truncation
        T_r[radius > self._R_truncation] = np.nan
        
        return radius, T_r.to('keV')

    
    #==================================================
    # Get the gas entropy profile
    #==================================================
    
    def get_entropy_gas_profile(self, radius=np.logspace(0,4,100)*u.kpc):
        """
        Get the entropy profile.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - K_r (quantity): the entropy profile in unit of keV cm2

        """

        # In case the input is not an array
        radius = model_tools.check_qarray(radius, unit='kpc')

        # Compute n and P
        radius, n_r = self.get_density_gas_profile(radius=radius)
        radius, P_r = self.get_pressure_gas_profile(radius=radius)

        # Get K
        n_r[n_r <= 0] = np.nan
        K_r = P_r / n_r**(5.0/3)

        # Apply truncation
        K_r[radius > self._R_truncation] = np.nan
        
        return radius, K_r.to('keV cm2')

    
    #==================================================
    # Get the hydrostatic mass profile
    #==================================================

    def get_hse_mass_profile(self, radius=np.logspace(0,4,100)*u.kpc):
        """
        Get the hydrostatic mass profile using exact analytical expressions.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        
        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - Mhse_r (quantity): the hydrostatic mass profile in unit of Msun

        """

        # In case the input is not an array
        radius = model_tools.check_qarray(radius, unit='kpc')
        
        #---------- Mean molecular weights
        mu_gas,mu_e,mu_p,mu_alpha = cluster_global.mean_molecular_weight(Y=self._helium_mass_fraction,
                                                                         Z=self._metallicity_sol*self._abundance)

        #---------- Get the electron density profile
        radius, n_r = self.get_density_gas_profile(radius=radius)

        #---------- Get dP/dr
        dpdr_r = self._get_generic_profile(radius, self._pressure_gas_model, derivative=True)
        dpdr_r[radius > self._R_truncation] *= 0

        #---------- Compute the mass
        n_r[n_r <= 0] = np.nan
        Mhse_r = -radius**2 / n_r * dpdr_r / (mu_gas*const.m_p*const.G)
        
        Mhse_r[radius > self._R_truncation] = np.nan
        
        return radius, Mhse_r.to('Msun')


    #==================================================
    # Get the overdensity contrast profile
    #==================================================

    def get_overdensity_contrast_profile(self, radius=np.logspace(0,4,100)*u.kpc):
        """
        Get the overdensity contrast profile.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        
        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - delta_r: the overdensity contrast profile

        """

        # In case the input is not an array
        radius = model_tools.check_qarray(radius, unit='kpc')

        # Compute delta from the mass profile
        r, mhse = self.get_hse_mass_profile(radius)
        rho_c = self._cosmo.critical_density(self._redshift)
        delta_r = mhse/(1.0-self._hse_bias) / (4.0/3.0*np.pi*radius**3 * rho_c)

        return radius, delta_r.to_value('')*u.adu


    #==================================================
    # Get the R500 profile
    #==================================================

    def get_mdelta_from_profile(self, delta=500, Rmin=10*u.kpc, Rmax=1e4*u.kpc):
        """
        Get R_delta and M_delta from the overdensity profile, given HSE equilibrium and HSE bias.
        
        Parameters
        ----------
        - delta : the overdensity considered, e.g. 2500, 500, 200
        - Rmin (quantity): the minimal range to search for Rdelta
        - Rmax (quantity): the maximal range to search for Rdelta
        
        Outputs
        ----------
        - Rdelta (quantity): the radius within which delta times the critical density is enclose, e.g. R500
        - Mdelta (quantity): the mass corresponding to Rdelta
        
        """

        # defines the function where to search for roots
        def overdensity_delta_difference(rkpc):
            rod, od = self.get_overdensity_contrast_profile(rkpc*u.kpc)
            func = od.to_value('adu') - delta
            func[np.isnan(od)] = -1.0 # Undefined overdensity (e.g. if truncation) should not be the root
            return func

        # Search the root
        Rdelta = brentq(overdensity_delta_difference, Rmin.to_value('kpc'), Rmax.to_value('kpc'))

        # In case the root is >= R_truncation, Rdelta was not reached
        if Rdelta >= self._R_truncation.to_value('kpc'):
            if not self._silent: print(('The truncation was reached before R'+str(delta)))
            Rdelta = np.nan

        # Get Mdelta as well
        Mdelta = cluster_global.Rdelta_to_Mdelta(Rdelta, self._redshift, delta=delta, cosmo=self._cosmo)
    
        return Rdelta*u.kpc, Mdelta*u.Msun


    #==================================================
    # Set b_hse given M500 and Mhse profile
    #==================================================

    def set_hsebias_from_m500_and_mhse(self):
        """
        Set the HSE mass bias given the mass parameter M500 and the Mhse profile.
        
        Parameters
        ----------
        
        Outputs
        ----------
        
        """
        
        M_HSE_R500 = self.get_hse_mass_profile(self._R500)[1][0]
        self._hse_bias = 1 - M_HSE_R500.to_value('Msun') / self._M500.to_value('Msun')


    #==================================================
    # Set M500 given b_hse and M_hse profile
    #==================================================

    def set_m500_from_bhse_and_mhse(self):
        """
        Set the value of M500 given the HSE bias and the HSE mass profile.
        
        Parameters
        ----------
        
        Outputs
        ----------
        
        """

        R500, M500 = self.get_mdelta_from_profile(delta=500, Rmin=10*u.kpc, Rmax=1e4*u.kpc)
        self.M500 = M500 # R500 set automatically
        
        
    #==================================================
    # Compute Mgas
    #==================================================
    def get_gas_mass_profile(self, radius=np.logspace(0,4,100)*u.kpc):
        """
        Get the gas mass profile.
        
        Parameters
        ----------
        - radius (quantity) : the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - Mgas_r (quantity): the gas mass profile 
        
        """

        # In case the input is not an array
        radius = model_tools.check_qarray(radius, unit='kpc')
        
        #---------- Mean molecular weights
        mu_gas,mu_e,mu_p,mu_alpha = cluster_global.mean_molecular_weight(Y=self._helium_mass_fraction,
                                                                         Z=self._metallicity_sol*self._abundance)
        
        #---------- Integrate the mass
        I_n_gas_r = np.zeros(len(radius))
        for i in range(len(radius)):
            # make sure we go well bellow rmax
            rmin = np.amin([self._Rmin.to_value('kpc'), radius.to_value('kpc')[i]/10.0])*u.kpc 
            rad = model_tools.sampling_array(rmin, radius[i], NptPd=self._Npt_per_decade_integ, unit=True)
            # To avoid ringing at Rtrunc, insert it if we are above
            if np.amax(rad) > self._R_truncation:
                rad = rad.insert(0, self._R_truncation)
                rad.sort()
            rad, n_r = self.get_density_gas_profile(radius=rad)
            I_n_gas_r[i] = model_tools.trapz_loglog(4*np.pi*rad**2*n_r, rad)
        
        Mgas_r = mu_e*const.m_p * I_n_gas_r

        return radius, Mgas_r.to('Msun')

    
    #==================================================
    # Compute fgas
    #==================================================

    def get_fgas_profile(self, radius=np.logspace(0,4,100)*u.kpc):
        """
        Get the gas fraction profile.
        
        Parameters
        ----------
        - radius : the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - fgas_r (quantity): the gas mass profile 

        """

        # In case the input is not an array
        radius = model_tools.check_qarray(radius, unit='kpc')
        
        # Compute fgas from Mgas and Mhse
        r, mgas = self.get_gas_mass_profile(radius)
        r, mhse = self.get_hse_mass_profile(radius)       
        fgas_r = mgas.to_value('Msun') / mhse.to_value('Msun') * (1.0 - self._hse_bias) 

        return radius, fgas_r*u.adu

    
    #==================================================
    # Compute thermal energy
    #==================================================

    def get_thermal_energy_profile(self, radius=np.logspace(0,4,100)*u.kpc):
        """
        Compute the thermal energy profile
        
        Parameters
        ----------
        - radius (quantity) : the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - Uth (quantity) : the thermal energy, in GeV

        """
        
        # In case the input is not an array
        radius = model_tools.check_qarray(radius, unit='kpc')
        
        #---------- Mean molecular weights
        mu_gas,mu_e,mu_p,mu_alpha = cluster_global.mean_molecular_weight(Y=self._helium_mass_fraction,
                                                                         Z=self._metallicity_sol*self._abundance)

        #---------- Integrate the pressure in 3d
        Uth_r = np.zeros(len(radius))*u.erg
        for i in range(len(radius)):
            # make sure we go well bellow rmax
            rmin = np.amin([self._Rmin.to_value('kpc'), radius.to_value('kpc')[i]/10.0])*u.kpc 
            rad = model_tools.sampling_array(rmin, radius[i], NptPd=self._Npt_per_decade_integ, unit=True)
            # To avoid ringing at Rtrunc, insert it if we are above
            if np.amax(rad) > self._R_truncation:
                rad = rad.insert(0, self._R_truncation)
                rad.sort()
            rad, p_r = self.get_pressure_gas_profile(radius=rad)
            Uth_r[i] = model_tools.trapz_loglog((3.0/2.0)*(mu_e/mu_gas) * 4*np.pi*rad**2 * p_r, rad)
                    
        return radius, Uth_r.to('erg')

    
    #==================================================
    # Get the gas electron density profile
    #==================================================
    
    def get_magfield_profile(self, radius=np.logspace(0,4,100)*u.kpc):
        """
        Get the magnetic field profile.
        
        Parameters
        ----------
        - radius (quantity) : the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - B_r (quantity): the magnetic field profile in unit of uG

        """

        # In case the input is not an array
        radius = model_tools.check_qarray(radius, unit='kpc')

        # get profile
        B_r = self._get_generic_profile(radius, self._magfield_model)
        B_r[radius > self._R_truncation] *= 0
        
        return radius, B_r.to('uG')

    
    #==================================================
    # Get normalized CR 2D spectrum X profile
    #==================================================
    
    def _get_normed_crp_2d(self, energy=np.logspace(-2,7,100)*u.GeV, radius=np.logspace(0,4,100)*u.kpc):
        """
        Compute the cosmic ray proton 2d distribution, but normalized.
        
        Parameters
        ----------
        - energy (quantity) : the physical energy of CR protons
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - distribution (quantity): in adu, with distribution[i_energy, i_radius]

        """

        # In case the input is not an array
        energy = model_tools.check_qarray(energy, unit='GeV')
        radius = model_tools.check_qarray(radius, unit='kpc')
        
        # get profile
        f_r = self._get_generic_profile(radius, self._density_crp_model)
        f_r[radius > self._R_truncation] *= 0
        f_r2 = model_tools.replicate_array(f_r.to_value('adu'), len(energy), T=False)

        # get spectrum
        f_E = self._get_generic_spectrum(energy, self._spectrum_crp_model)
        f_E[energy > self._Epmax] *= 0
        f_E[energy < self._Epmin] *= 0
        f_E2 = model_tools.replicate_array(f_E, len(radius), T=True)

        # compute the distrib
        distribution = f_E2 * f_r2

        return distribution * u.adu

    
    def _get_normed_cre1_2d(self, energy=np.logspace(-2,7,100)*u.GeV, radius=np.logspace(0,4,100)*u.kpc):
        """
        Compute the cosmic ray primary electron 2d distribution, but normalized.
        
        Parameters
        ----------
        - energy (quantity) : the physical energy of CR electrons
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - distribution (quantity): in unit of GeV-1 cm-3, with spectrum[i_energy, i_radius]

        """

        # In case the input is not an array
        energy = model_tools.check_qarray(energy, unit='GeV')
        radius = model_tools.check_qarray(radius, unit='kpc')

        # In case losses are applied, energy range should extend because it is integrated up to infinity
        if self._cre1_loss_model is not 'None':
            emin = np.amax([(const.m_e*const.c**2).to_value('GeV'),
                            np.amin(energy.to_value('GeV'))])*u.GeV # min of CRe energy requested, or m_e c^2
            emax = self._Eemax
            eng = model_tools.sampling_array(emin, emax, NptPd=self._Npt_per_decade_integ, unit=True)
            rad = copy.copy(radius)
        else:
            # The distribution does not need integration, keep initial requested energy and radius
            eng = copy.copy(energy)
            rad = copy.copy(radius)
            
        # get profile
        f_r = self._get_generic_profile(rad, self._density_cre1_model)
        f_r[rad > self._R_truncation] *= 0 
        f_r2 = model_tools.replicate_array(f_r.to_value('adu'), len(eng), T=False)
        
        # get spectrum
        f_E = self._get_generic_spectrum(eng, self._spectrum_cre1_model)
        f_E[eng > self._Eemax] *= 0
        f_E[eng < self._Eemin] *= 0
        f_E2 = model_tools.replicate_array(f_E, len(rad), T=True)
    
        # compute the distrib
        distribution = f_E2 * f_r2

        # Apply losses to the distribution if needed
        if self._cre1_loss_model is not 'None':
            if self._cre1_loss_model == 'Steady':
                distribution = self.apply_steady_state_electron_loss(energy, radius,
                                                                     distribution*u.adu, eng, rad)
                distribution = distribution.value
            if self._cre1_loss_model == 'Injection':
                print('Not implemented, keep the initial distribution')

        return distribution * u.adu
    
    
    #==================================================
    # Apply steady state losses to a given injection rate
    #==================================================
     
    def apply_steady_state_electron_loss(self, energy, radius,
                                         dN_dEdVdt, eng, rad):
        """
        Apply steady state solution to get dN/dEdV from dN/dEdVdt
        The steady state solution is used:  dN/dEdV(E,r) = 1/L(E,r) * \int_E^\infty Q(E) dE
        See e.g. Sarrazin (1999) eq 38.
        
        Parameters
        ----------
        - energy (quantity) : the physical energy of the output
        - radius (quantity): the physical 3d radius of the output
        - dN_dEdVdt (quantity): the rate of injection
        - eng (quantity): the energy associated to dN_dEdVdt
        - rad (quantity): the radius associated to dN_dEdVdt

        Outputs
        ----------
        - distribution (quantity): in adu, with distribution[i_energy, i_radius]

        """
        
        # In case the input is not an array
        energy = model_tools.check_qarray(energy, unit='GeV')
        radius = model_tools.check_qarray(radius, unit='kpc')

        # Get the necessary quantity
        radius, n_e = self.get_density_gas_profile(radius)
        radius, B   = self.get_magfield_profile(radius)
        
        # Get the losses
        mu_gas,mu_e,mu_p,mu_alpha = cluster_global.mean_molecular_weight(Y=self._helium_mass_fraction,
                                                                         Z=self._metallicity_sol*self._abundance)
        dEdt = cluster_electron_loss.dEdt_tot(energy, radius=radius, n_e=n_e, B=B, redshift=self._redshift,
                                              ne2nth=mu_e/mu_p+mu_e/mu_alpha)

        # Compute the integral \int_E^\infty Q(E) dE
        dN_dEdVdt_integrated = np.zeros((len(energy),len(radius)))* dN_dEdVdt.unit*eng.unit
        eng_grid = model_tools.replicate_array(eng, len(rad), T=True) # eng, but replicated along rad
        
        for i in range(len(energy)):
            dN_dEdVdt_i = dN_dEdVdt.copy()
            dN_dEdVdt_i[eng_grid < energy[i]] *= 0
            dN_dEdVdt_integrated[i,:] = model_tools.trapz_loglog(dN_dEdVdt_i, eng, axis=0)
        
        # Compute the solution the equation: dN/dEdV(E,r) = 1/L(E,r) * \int_E^\infty Q(E) dE
        energy_grid = model_tools.replicate_array(energy, len(radius), T=True)
        w_neg = energy_grid <= max(const.m_e*const.c**2, self._Eemin) # flag energies that are not possible
        dEdt[w_neg] = 1*dEdt.unit
        dN_dEdV = dN_dEdVdt_integrated / dEdt
        dN_dEdV[w_neg] = 0
        
        return dN_dEdV
    
    
    #==================================================
    # Get the CR proton normalization
    #==================================================

    def _get_crp_normalization(self):
        """
        Compute the normalization of the cosmic ray proton distribution:
        dN/dE/dV = Norm f(E) f(r)
        with f(E) the spectral form and f(r) the spatial form
        Norm is in unit of GeV-1 cm-3
        
        Parameters
        ----------

        Outputs
        ----------
        - Norm (quantity): in unit of GeV-1 cm-3

        """

        Rcut = self._X_crp_E['R_norm']
        
        # Get the thermal energy
        rad_uth, U_th = self.get_thermal_energy_profile(Rcut)
        
        # Get the 2d distribution
        rad = model_tools.sampling_array(self._Rmin, Rcut, NptPd=self._Npt_per_decade_integ, unit=True)
        eng = model_tools.sampling_array(self._Epmin, self._Epmax, NptPd=self._Npt_per_decade_integ, unit=True)
        f_cr_e_r = self._get_normed_crp_2d(eng, rad)

        # Integrate over radius then energy
        rad2d = model_tools.replicate_array(rad, len(eng), T=False)
        f_cr_e = model_tools.trapz_loglog(4*np.pi*rad2d**2 * f_cr_e_r.to_value('adu'), rad2d, axis=1)
        Icr = model_tools.trapz_loglog(eng * f_cr_e, eng)
        
        # Compute the normalization
        Norm = self._X_crp_E['X'] * U_th / Icr

        return Norm.to('GeV-1 cm-3')
    

    #==================================================
    # Get the CR primary electron normalization
    #==================================================
    
    def _get_cre1_normalization(self):
        """
        Compute the normalization of the cosmic ray primary electron distribution:
        dN/dE/dV = Norm f(E) f(r)
        with f(E) the spectral form and f(r) the spatial form
        Norm is in unit of GeV-1 cm-3
        
        Parameters
        ----------

        Outputs
        ----------
        - Norm (quantity): in unit of GeV-1 cm-3

        """

        Rcut = self._X_cre1_E['R_norm']
        
        # Get the thermal energy
        rad_uth, U_th = self.get_thermal_energy_profile(Rcut)
        
        # Get the 2d distribution
        rad = model_tools.sampling_array(self._Rmin, Rcut, NptPd=self._Npt_per_decade_integ, unit=True)
        eng = model_tools.sampling_array(self._Eemin, self._Eemax, NptPd=self._Npt_per_decade_integ, unit=True)
        f_cr_e_r = self._get_normed_cre1_2d(eng, rad)

        # Integrate over radius then energy
        rad2d = model_tools.replicate_array(rad, len(eng), T=False)
        f_cr_e = model_tools.trapz_loglog(4*np.pi*rad2d**2 * f_cr_e_r.to_value('adu'), rad2d, axis=1)
        Icr = model_tools.trapz_loglog(eng * f_cr_e, eng)
        
        # Compute the normalization
        Norm = self._X_cre1_E['X'] * U_th / Icr

        return Norm.to('GeV-1 cm-3')
    
    
    #==================================================
    # Get the CR proton 2d distribution
    #==================================================
    
    def get_crp_2d(self, energy=np.logspace(-2,7,100)*u.GeV, radius=np.logspace(0,4,100)*u.kpc):
        """
        Compute the cosmic ray proton 2d distribution.
        
        Parameters
        ----------
        - energy (quantity) : the physical energy of CR protons
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - distribution (quantity): in unit of GeV-1 cm-3, with distribution[i_energy, i_radius]

        """
        
        norm = self._get_crp_normalization()
        distribution = norm * self._get_normed_crp_2d(energy, radius).to_value('adu')

        return distribution.to('GeV-1 cm-3')


    #==================================================
    # Get the CR primary electron 2d distribution
    #==================================================

    def get_cre1_2d(self, energy=np.logspace(-2,7,100)*u.GeV, radius=np.logspace(0,4,100)*u.kpc):
        """
        Compute the cosmic ray primary electron 2d distribution.
        
        Parameters
        ----------
        - energy (quantity) : the physical energy of CR protons
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - distribution (quantity): in unit of GeV-1 cm-3, with distribution[i_energy, i_radius]

        """

        norm = self._get_cre1_normalization()
        distribution = norm * self._get_normed_cre1_2d(energy, radius).to_value('adu')
     
        return distribution.to('GeV-1 cm-3')
    
    
    #==================================================
    # Get the CR proton density profile
    #==================================================
    
    def get_density_crp_profile(self, radius=np.logspace(0,4,100)*u.kpc,
                                Emin=None, Emax=None, Energy_density=False):
        """
        Compute the cosmic ray proton density profile, integrating energies 
        between Emin and Emax.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        - Emin (quantity): the lower bound for energy integration
        - Emax (quantity): the upper bound for energy integration
        - Energy_density (bool): if True, then the energy density is computed. Otherwise, 
        the number density is computed.

        Outputs
        ----------
        - density (quantity): in unit of cm-3 or GeV cm-3

        """

        # In case the input is not an array
        radius = model_tools.check_qarray(radius, unit='kpc')

        # Define energy
        if Emin is None:
            Emin = self._Epmin
        if Emax is None:
            Emax = self._Epmax        
            
        # Integrate over the spectrum
        eng = model_tools.sampling_array(Emin, Emax, NptPd=self._Npt_per_decade_integ, unit=True)
        dN_dEdV = self.get_crp_2d(eng, radius)

        if Energy_density:
            profile = (model_tools.trapz_loglog(np.vstack(eng.to_value('GeV'))*u.GeV * dN_dEdV,
                                                eng, axis=0)).to('GeV cm-3')            
        else:
            profile = (model_tools.trapz_loglog(dN_dEdV, eng, axis=0)).to('cm-3')
            
        return radius, profile
    
    
    #==================================================
    # Get the CR primary electron density profile
    #==================================================
    
    def get_density_cre1_profile(self, radius = np.logspace(0,4,100)*u.kpc, 
                                 Emin = None, Emax = None, Energy_density = False):
        """
        Compute the cosmic ray primary electron density profile, integrating energies 
        between Emin and Emax.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        - Emin (quantity): the lower bound for energy integration
        - Emax (quantity): the upper bound for energy integration
        - Energy_density (bool): if True, then the energy density is computed. Otherwise, 
        the number density is computed.

        Outputs
        ----------
        - density (quantity): in unit of cm-3 or GeV cm-3

        """
        
        #In case the input is not an array
        radius = model_tools.check_qarray(radius, unit = 'kpc')

        #Define energy
        if Emin is None:
            Emin = self._Eemin
        if Emax is None:
            Emax = self._Eemax
    
        #Define energy range
        eng = model_tools.sampling_array(Emin, Emax,NptPd = self._Npt_per_decade_integ, unit = True)
    
        #2D differntial spectrum
        dN_dEdV = self.get_cre1_2d( eng, radius)
    
        #Integrate the diffferential spectrum 
        if Energy_density:
            profile = (model_tools.trapz_loglog(np.vstack(eng.to_value('GeV'))*u.GeV * dN_dEdV,
                                                eng , axis =0)).to('GeV cm-3')

        else:
            profile = (model_tools.trapz_loglog(dN_dEdV, eng, axis = 0)).to('cm-3')
    
        return radius, profile


    #==================================================
    # Get the CR proton spectrum
    #==================================================
    
    def get_crp_spectrum(self, energy=np.logspace(-2,7,100)*u.GeV, Rmax=None):
        """
        Compute the cosmic ray proton spectrum, integrating radius 
        between 0 and Rmax.
        
        Parameters
        ----------
        - energy (quantity) : the physical energy of CR protons
        - Rmax (quantity): the radius within with the spectrum is computed 
        (default is R500)

        Outputs
        ----------
        - spectrum (quantity): in unit of GeV-1

        """

        # In case the input is not an array
        energy = model_tools.check_qarray(energy, unit='GeV')

        # define the maximum radius
        if Rmax is None:
            Rmax = self._R500
                
        # Define the radius for integration
        rmin = np.amin([self._Rmin.to_value('kpc'), Rmax.to_value('kpc')/10])*u.kpc # make sure we go low enough
        rad = model_tools.sampling_array(rmin, Rmax, NptPd=self._Npt_per_decade_integ, unit=True)

        # To improve precision around R_truncation in integration
        if np.amax(rad) > self._R_truncation:
            rad = rad.insert(0, self._R_truncation)
            rad.sort()

        # Get the differential spectrum/profile
        dN_dEdV = self.get_crp_2d(energy, rad)
        
        # Integrate
        spectrum = model_tools.trapz_loglog(4*np.pi*rad**2 * dN_dEdV, rad)

        return energy, spectrum.to('GeV-1')
    

    #==================================================
    # Get the CR primary electron spectrum
    #==================================================

    def get_cre1_spectrum(self, energy=np.logspace(-2,7,100)*u.GeV, Rmax=None):
        """
        Compute the cosmic ray primary electron spectrum, integrating radius 
        between 0 and Rmax.
        
        Parameters
        ----------
        - energy (quantity) : the physical energy of CR primary electrons
        - Rmax (quantity): the radius within with the spectrum is computed 
        (default is R500)

        Outputs
        ----------
        - spectrum (quantity): in unit of GeV-1

        """   
        # In case the input is not an array
        energy = model_tools.check_qarray(energy, unit='GeV')

        # define the maximum radius
        if Rmax is None:
            Rmax = self._R500
                
        # Define the radius for integration
        rmin = np.amin([self._Rmin.to_value('kpc'), Rmax.to_value('kpc')/10])*u.kpc 
        rad = model_tools.sampling_array(rmin, Rmax, NptPd= self._Npt_per_decade_integ, unit=True)

        if np.amax(rad) > self._R_truncation:
            rad = rad.insert(0, self._R_truncation)
            rad.sort()

        # Get the differential spectrum/profile
        dN_dEdV = self.get_cre1_2d(energy, rad)
  
        # Integrate
        spectrum = model_tools.trapz_loglog(4*np.pi*rad**2 * dN_dEdV, rad)

        return energy, spectrum.to('GeV-1')

    
    #==================================================
    # Get the CR to thermal energy profile
    #==================================================
    
    def get_crp_to_thermal_energy_profile(self, radius=np.logspace(0,4,100)*u.kpc,
                                           Emin=None, Emax=None):
        """
        Compute the X_crp_E profile, i.e. the cosmic ray to thermal energy enclosed within R
        profile.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        - Emin (quantity): the lower bound for energy integration
        - Emax (quantity): the upper bound for energy integration

        Outputs
        ----------
        - x_r (np.ndarray): the profile

        """

        # In case the input is not an array
        radius = model_tools.check_qarray(radius, unit='kpc')

        # Define energy (going beyond Epmin/Epmax only introduces more integration error because of discountinuity)
        if Emin is None or Emin < self._Epmin:
            Emin = self._Epmin
        if Emax is None or Emax > self._Epmax:
            Emax = self._Epmax
        
        # Thermal energy
        r_uth, Uth_r = self.get_thermal_energy_profile(radius)

        # Integrate CR energy density profile
        Ucr_r = np.zeros(len(radius))*u.GeV
        for i in range(len(radius)):
            # make sure we go well bellow rmax
            rmin = np.amin([self._Rmin.to_value('kpc'), radius.to_value('kpc')[i]/10.0])*u.kpc 
            rad = model_tools.sampling_array(rmin, radius[i], NptPd=self._Npt_per_decade_integ, unit=True)
            # To avoid ringing at Rtrunc, insert it if we are above
            if np.amax(rad) > self._R_truncation:
                rad = rad.insert(0, self._R_truncation)
                rad.sort()
            rad, e_cr = self.get_density_crp_profile(rad, Emin=Emin, Emax=Emax, Energy_density=True)
            Ucr_r[i] = model_tools.trapz_loglog(4*np.pi*rad**2 * e_cr, rad)

        # X(<R)
        x_r = Ucr_r.to_value('GeV') / Uth_r.to_value('GeV')

        return radius, x_r*u.adu


    #==================================================
    # Get the CRe1 to thermal energy profile
    #==================================================
    
    def get_cre1_to_thermal_energy_profile(self, radius=np.logspace(0,4,100)*u.kpc,
                                           Emin=None, Emax=None):
        """
        Compute the X_cre1_E profile, i.e. the primary cosmic ray electron to thermal 
        energy enclosed within R profile.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        - Emin (quantity): the lower bound for energy integration
        - Emax (quantity): the upper bound for energy integration

        Outputs
        ----------
        - x_r (np.ndarray): the profile

        """

        # In case the input is not an array
        radius = model_tools.check_qarray(radius, unit='kpc')

        # Define energy (going beyond Eemin/Eemax only introduces more integration error because of discountinuity)
        if Emin is None or Emin < self._Eemin:
            Emin = self._Eemin
        if Emax is None or Emax > self._Eemax:
            Emax = self._Eemax
        
        # Thermal energy
        r_uth, Uth_r = self.get_thermal_energy_profile(radius)

        # Integrate CR energy density profile
        Ucr_r = np.zeros(len(radius))*u.GeV
        for i in range(len(radius)):
            # make sure we go well bellow rmax
            rmin = np.amin([self._Rmin.to_value('kpc'), radius.to_value('kpc')[i]/10.0])*u.kpc 
            rad = model_tools.sampling_array(rmin, radius[i], NptPd=self._Npt_per_decade_integ, unit=True)
            # To avoid ringing at Rtrunc, insert it if we are above
            if np.amax(rad) > self._R_truncation:
                rad = rad.insert(0, self._R_truncation)
                rad.sort()
            rad, e_cr = self.get_density_cre1_profile(rad, Emin=Emin, Emax=Emax, Energy_density=True)
            Ucr_r[i] = model_tools.trapz_loglog(4*np.pi*rad**2 * e_cr, rad)

        # X(<R)
        x_r = Ucr_r.to_value('GeV') / Uth_r.to_value('GeV')

        return radius, x_r*u.adu

    
    #==================================================
    # Get the CRe1 to thermal energy profile
    #==================================================
    
    def get_cre2_to_thermal_energy_profile(self, radius=np.logspace(0,4,100)*u.kpc,
                                           Emin=None, Emax=None):
        """
        Compute the X_cre2_E profile, i.e. the secondary cosmic ray electron to thermal 
        energy enclosed within R profile.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        - Emin (quantity): the lower bound for energy integration
        - Emax (quantity): the upper bound for energy integration

        Outputs
        ----------
        - x_r (np.ndarray): the profile

        """

        # In case the input is not an array
        radius = model_tools.check_qarray(radius, unit='kpc')

        # Define energy
        if Emin is None:
            Emin = (const.m_e*const.c**2).to('GeV')
        if Emax is None:
            Emax = self._Epmax
            
        # Thermal energy
        r_uth, Uth_r = self.get_thermal_energy_profile(radius)

        # Integrate CR energy density profile
        Ucr_r = np.zeros(len(radius))*u.GeV
        for i in range(len(radius)):
            # make sure we go well bellow rmax
            rmin = np.amin([self._Rmin.to_value('kpc'), radius.to_value('kpc')[i]/10.0])*u.kpc 
            rad = model_tools.sampling_array(rmin, radius[i], NptPd=self._Npt_per_decade_integ, unit=True)
            # To avoid ringing at Rtrunc, insert it if we are above
            if np.amax(rad) > self._R_truncation:
                rad = rad.insert(0, self._R_truncation)
                rad.sort()
            rad, e_cr = self.get_density_cre2_profile(rad, Emin=Emin, Emax=Emax, Energy_density=True)
            Ucr_r[i] = model_tools.trapz_loglog(4*np.pi*rad**2 * e_cr, rad)

        # X(<R)
        x_r = Ucr_r.to_value('GeV') / Uth_r.to_value('GeV')

        return radius, x_r*u.adu
    
    
    #==================================================
    # Get the gamma ray production rate
    #==================================================
    
    def get_rate_gamma(self, energy=np.logspace(-2,7,100)*u.GeV, radius=np.logspace(0,4,100)*u.kpc,
                       model='Kafexhiu2014', limit='mixed'):
        """
        Compute the gamma ray production rate as dN/dEdVdt = f(E, r)
        
        Parameters
        ----------
        - energy (quantity) : the physical energy of gamma rays
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        - model (str): change the reference model to 'Kafexhiu2014' or 'Kelner2006'
        - limit (str): in which limit to compute the result in the case of Kelner2006

        Outputs
        ----------
        - dN_dEdVdt (np.ndarray): the differntial production rate

        """
        
        # In case the input is not an array
        energy = model_tools.check_qarray(energy, unit='GeV')
        radius = model_tools.check_qarray(radius, unit='kpc')
        
        # Get the thermal proton density profile
        mu_gas,mu_e,mu_p,mu_alpha = cluster_global.mean_molecular_weight(Y=self._helium_mass_fraction,
                                                                         Z=self._metallicity_sol*self._abundance)
        rad, n_e  = self.get_density_gas_profile(radius)
        n_H = n_e * mu_e/mu_p

        # Parse the CRp distribution: returns call function[rad, energy] amd returns f[rad, energy]
        def Jp(rad, eng): return self.get_crp_2d(eng*u.GeV, rad*u.kpc).value.T
        # Case of K14
        if model == 'Kafexhiu2014':
            model = K14.PPmodel(Jp,
                                Y0=self._helium_mass_fraction,
                                Z0=self._metallicity_sol,
                                abundance=self._abundance,
                                hiEmodel=self._pp_interaction_model,
                                Epmin=self._Epmin,
                                Epmax=self._Epmax,
                                NptEpPd=self._Npt_per_decade_integ)

            dN_dEdVdt = model.gamma_spectrum(energy, radius, n_H).T
        # Case of K14
        elif model == 'Kelner2006':
            model = K06.PPmodel(Jp,
                                Epmin=self._Epmin,
                                Epmax=self._Epmax,
                                NptEpPd=self._Npt_per_decade_integ)
            
            dN_dEdVdt = model.gamma_spectrum(energy, radius, n_H, limit=limit).T
        # Error
        else:
            raise ValueError('Only Kafexhiu2014 and Kelner2006 models are available.')

        return dN_dEdVdt.to('GeV-1 cm-3 s-1')


    #==================================================
    # Get the secondary electron production rate
    #==================================================
    
    def get_rate_cre2(self, energy=np.logspace(-2,7,100)*u.GeV, radius=np.logspace(0,4,100)*u.kpc,
                      model='Kafexhiu2014', limit='mixed'):
        """
        Compute the cosmic ray electron production rate as dN/dEdVdt = f(E, r)
        
        Note
        ----
        At high energy, wit few point per decade, some "lines" may appear in the spectrum
        due to numerical issues.

        Parameters
        ----------
        - energy (quantity) : the physical energy of electrons
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        - model (str): change the reference model to 'Kafexhiu2014' or 'Kelner2006'
        - limit (str): in which limit to compute the result in the case of Kelner2006

        Outputs
        -------
        - dN_dEdVdt (np.ndarray): the differntial production rate

        """
        
        # In case the input is not an array
        energy = model_tools.check_qarray(energy, unit='GeV')
        radius = model_tools.check_qarray(radius, unit='kpc')
        
        # Get the thermal proton density profile
        mu_gas,mu_e,mu_p,mu_alpha = cluster_global.mean_molecular_weight(Y=self._helium_mass_fraction,
                                                                         Z=self._metallicity_sol*self._abundance)
        rad, n_e  = self.get_density_gas_profile(radius)
        n_H = n_e * mu_e/mu_p

        # Parse the CRp distribution: returns call function[rad, energy] amd returns f[rad, energy]
        def Jp(rad, eng): return self.get_crp_2d(eng*u.GeV, rad*u.kpc).value.T
        
        # Case of K14
        if model == 'Kafexhiu2014':
            model = K14.PPmodel(Jp,
                                Y0=self._helium_mass_fraction,
                                Z0=self._metallicity_sol,
                                abundance=self._abundance,
                                hiEmodel=self._pp_interaction_model,
                                Epmin=self._Epmin,
                                Epmax=self._Epmax,
                                NptEpPd=self._Npt_per_decade_integ)

            dN_dEdVdt = model.electron_spectrum(energy, radius, n_H).T

        # Case of K14
        elif model == 'Kelner2006':
            model = K06.PPmodel(Jp,
                                Epmin=self._Epmin,
                                Epmax=self._Epmax,
                                NptEpPd=self._Npt_per_decade_integ)
            
            dN_dEdVdt = model.electron_spectrum(energy, radius, n_H, limit=limit).T

        # Error
        else:
            raise ValueError('Only Kafexhiu2014 and Kelner2006 models are available.')

        return dN_dEdVdt.to('GeV-1 cm-3 s-1')


    #==================================================
    # Get the neutrino production rate
    #==================================================
    
    def get_rate_neutrino(self, energy=np.logspace(-2,7,100)*u.GeV, radius=np.logspace(0,4,100)*u.kpc,
                          model='Kafexhiu2014', limit='mixed', flavor='all'):
        """
        Compute the cosmic ray electron production rate as dN/dEdVdt = f(E, r)
        
        Note
        ----
        At high energy, wit few point per decade, some "lines" may appear in the spectrum
        due to numerical issues.

        Parameters
        ----------
        - energy (quantity) : the physical energy of neutrinos
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        - model (str): change the reference model to 'Kafexhiu2014' or 'Kelner2006'
        - limit (str): in which limit to compute the result in the case of Kelner2006
        - flavor (str): either 'all', 'numu' or 'nue'

        Outputs
        -------
        - dN_dEdVdt (np.ndarray): the differntial production rate

        """
        
        # In case the input is not an array
        energy = model_tools.check_qarray(energy, unit='GeV')
        radius = model_tools.check_qarray(radius, unit='kpc')
        
        # Get the thermal proton density profile
        mu_gas,mu_e,mu_p,mu_alpha = cluster_global.mean_molecular_weight(Y=self._helium_mass_fraction,
                                                                         Z=self._metallicity_sol*self._abundance)
        rad, n_e  = self.get_density_gas_profile(radius)
        n_H = n_e * mu_e/mu_p

        # Parse the CRp distribution: returns call function[rad, energy] amd returns f[rad, energy]
        def Jp(rad, eng): return self.get_crp_2d(eng*u.GeV, rad*u.kpc).to_value('GeV-1 cm-3').T

        # Case of K14
        if model == 'Kafexhiu2014':
            model = K14.PPmodel(Jp,
                                Y0=self._helium_mass_fraction,
                                Z0=self._metallicity_sol,
                                abundance=self._abundance,
                                hiEmodel=self._pp_interaction_model,
                                Epmin=self._Epmin,
                                Epmax=self._Epmax,
                                NptEpPd=self._Npt_per_decade_integ)

            # Extract the spectrum
            if flavor == 'all':
                dN_dEdVdt1 = model.neutrino_spectrum(energy, radius, n_H, flavor='numu').T
                dN_dEdVdt2 = model.neutrino_spectrum(energy, radius, n_H, flavor='nue').T
                dN_dEdVdt = dN_dEdVdt1 + dN_dEdVdt2
                
            elif flavor == 'numu':
                dN_dEdVdt = model.neutrino_spectrum(energy, radius, n_H, flavor='numu').T
                
            elif flavor == 'nue':
                dN_dEdVdt = model.neutrino_spectrum(energy, radius, n_H, flavor='nue').T
                
            else :
                raise ValueError('Only all, numu and nue flavor are available.')
        
        # Case of K14
        elif model == 'Kelner2006':
            model = K06.PPmodel(Jp,
                                Epmin=self._Epmin,
                                Epmax=self._Epmax,
                                NptEpPd=self._Npt_per_decade_integ)

            # Extract the spectrum
            if flavor == 'all':
                dN_dEdVdt1 = model.neutrino_spectrum(energy, radius, n_H, limit=limit, flavor='numu').T
                dN_dEdVdt2 = model.neutrino_spectrum(energy, radius, n_H, limit=limit, flavor='nue').T
                dN_dEdVdt = dN_dEdVdt1 + dN_dEdVdt2
                
            elif flavor == 'numu':
                dN_dEdVdt = model.neutrino_spectrum(energy, radius, n_H, limit=limit, flavor='numu').T
                
            elif flavor == 'nue':
                dN_dEdVdt = model.neutrino_spectrum(energy, radius, n_H, limit=limit, flavor='nue').T
                
            else :
                raise ValueError('Only all, numu and nue flavor are available.')

        # Error
        else:
            raise ValueError('Only Kafexhiu2014 and Kelner2006 models are available.')

        return dN_dEdVdt.to('GeV-1 cm-3 s-1')


    #==================================================
    # Get the secondary electron spectrum
    #==================================================
    
    def get_cre2_2d(self, energy=np.logspace(-2,7,100)*u.GeV, radius=np.logspace(0,4,100)*u.kpc):
        """
        Compute the electron spectrum as dN/dEdV = f(E, r)
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        - energy (quantity) : the physical energy of electrons

        Outputs
        -------
        - dN_dEdV (np.ndarray): the number of electrons per unit volume and energy

        """
        
        # In case the input is not an array
        energy = model_tools.check_qarray(energy, unit='GeV')
        radius = model_tools.check_qarray(radius, unit='kpc')

        # Get the injection rate between the min and max possible, i.e. Epmax
        emin = np.amax([(const.m_e*const.c**2).to_value('GeV'),
                        np.amin(energy.to_value('GeV'))])*u.GeV # min of CRe energy requested, or m_e c^2
        emax = self._Epmax
        eng = model_tools.sampling_array(emin, emax, NptPd=self._Npt_per_decade_integ, unit=True)
        dN_dEdVdt = self.get_rate_cre2(eng, radius) # This is the time consumming part

        # Apply the losses
        dN_dEdV = self.apply_steady_state_electron_loss(energy, radius, dN_dEdVdt, eng, radius)

        return dN_dEdV.to('GeV-1 cm-3')


    #==================================================
    # Get the CR secondary electron density profile
    #==================================================
    
    def get_density_cre2_profile(self, radius=np.logspace(0,4,100)*u.kpc,
                                 Emin=None, Emax=None, Energy_density=False):
        """
        Compute the cosmic ray electron density profile, integrating energies 
        between Emin and Emax.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        - Emin (quantity): the lower bound for energy integration
        - Emax (quantity): the upper bound for energy integration
        - Energy_density (bool): if True, then the energy density is computed. Otherwise, 
        the number density is computed.

        Outputs
        ----------
        - density (quantity): in unit of cm-3 or GeV cm-3

        """

        # In case the input is not an array
        radius = model_tools.check_qarray(radius, unit='kpc')

        # Define energy
        if Emin is None:
            Emin = (const.m_e*const.c**2).to('GeV')
        if Emax is None:
            Emax = self._Epmax
            
        # Integrate over the spectrum
        eng = model_tools.sampling_array(Emin, Emax, NptPd=self._Npt_per_decade_integ, unit=True)
        dN_dEdV = self.get_cre2_2d(eng, radius)

        if Energy_density:
            profile = (model_tools.trapz_loglog(np.vstack(eng.to_value('GeV'))*u.GeV * dN_dEdV,
                                                eng, axis=0)).to('GeV cm-3')            
        else:
            profile = (model_tools.trapz_loglog(dN_dEdV, eng, axis=0)).to('cm-3')
            
        return radius, profile


    #==================================================
    # Get the CR secondary spectrum
    #==================================================
    
    def get_cre2_spectrum(self, energy=np.logspace(-2,7,100)*u.GeV, Rmax=None):
        """
        Compute the cosmic ray proton spectrum, integrating radius 
        between 0 and Rmax.
        
        Parameters
        ----------
        - energy (quantity) : the physical energy of CR protons
        - Rmax (quantity): the radius within with the spectrum is computed 
        (default is R500)

        Outputs
        ----------
        - spectrum (quantity): in unit of GeV-1

        """

        # In case the input is not an array
        energy = model_tools.check_qarray(energy, unit='GeV')

        # Define the radius for integration
        if Rmax is None:
            Rmax = self._R500
                
        # Integrate over the considered volume
        rmin = np.amin([self._Rmin.to_value('kpc'), Rmax.to_value('kpc')/10])*u.kpc # make sure we go low enough
        rad = model_tools.sampling_array(rmin, Rmax, NptPd=self._Npt_per_decade_integ, unit=True)

        # To improve precision around R_truncation in integration
        if np.amax(rad) > self._R_truncation:
            rad = rad.insert(0, self._R_truncation)
            rad.sort()

        # Get the differential spectrum/profile
        dN_dEdV = self.get_cre2_2d(energy, rad)

        # Integrate
        spectrum = model_tools.trapz_loglog(4*np.pi*rad**2 * dN_dEdV, rad)

        return energy, spectrum.to('GeV-1')
    
    
    #==================================================
    # Get the synchrotron spectrum
    #==================================================
    
    def get_rate_synchrotron(self, energy=np.logspace(-9,-2,100)*u.eV, radius=np.logspace(0,4,100)*u.kpc):
        """
        Compute the synchrotron density as dN/dEdVdt = f(E, r)
        
        Parameters
        ----------
        - energy (quantity) : the physical energy of synchrotron photons
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - dN_dEdVdt (np.ndarray): the differntial production rate

        """

        # In case the input is not an array
        energy = model_tools.check_qarray(energy, unit='eV')
        radius = model_tools.check_qarray(radius, unit='kpc')

        # Get the magnetic field
        radius, B   = self.get_magfield_profile(radius)
        
        # Parse the CRe distribution: returns call function[rad, energy] amd returns f[rad, energy]
        def Je(rad, eng):
            if self._X_crp_E['X'] == 0: 
                Sec_e = 0
            else: 
                Sec_e = self.get_cre2_2d(eng*u.GeV, rad*u.kpc).to_value('GeV-1 cm-3').T
            if self._X_cre1_E['X'] == 0: 
                Pri_e = 0
            else: 
                Pri_e = self.get_cre1_2d(eng*u.GeV, rad*u.kpc).to_value('GeV-1 cm-3').T           	
            return Pri_e + Sec_e

        # Define the model
        model = cluster_electron_emission.ClusterElectronEmission(Je,
                                                                  Eemin=(const.m_e*const.c**2).to('GeV'),
                                                                  Eemax=self._Epmax,
                                                                  NptEePd=self._Npt_per_decade_integ)
        
        # Extract the spectrum: what is long is evaluating Je inside the code
        dN_dEdVdt = model.synchrotron(energy, radius_input=radius, B=B).T
        
        return dN_dEdVdt.to('GeV-1 cm-3 s-1')


    #==================================================
    # Get the IC spectrum
    #==================================================
    
    def get_rate_ic(self, energy=np.logspace(-2,7,100)*u.GeV, radius=np.logspace(0,4,100)*u.kpc):
        """
        Compute the inverse compton density as dN/dEdVdt = f(E, r)
        
        Note
        ----------
        At high energy, the IC emission analytical parametrization present sharp features 
        which require a rather high NptEePD (10 is clearly to low and will induce wiggles 
        in the spectrum)

        Parameters
        ----------
        - energy (quantity) : the physical energy of photons
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array

        Outputs
        ----------
        - dN_dEdVdt (np.ndarray): the differntial production rate

        """

        # In case the input is not an array
        energy = model_tools.check_qarray(energy, unit='GeV')
        radius = model_tools.check_qarray(radius, unit='kpc')

        # Parse the CRe distribution: returns call function[rad, energy] amd returns f[rad, energy]
        def Je(rad, eng):
            if self._X_crp_E['X'] == 0: 
                Sec_e = 0
            else: 
                Sec_e = self.get_cre2_2d(eng*u.GeV, rad*u.kpc).to_value('GeV-1 cm-3').T
            if self._X_cre1_E['X'] == 0: 
                Pri_e = 0
            else: 
                Pri_e = self.get_cre1_2d(eng*u.GeV, rad*u.kpc).to_value('GeV-1 cm-3').T           	
            return Pri_e + Sec_e

        # Define the model
        model = cluster_electron_emission.ClusterElectronEmission(Je,
                                                                  Eemin=(const.m_e*const.c**2).to('GeV'),
                                                                  Eemax=self._Epmax,
                                                                  NptEePd=self._Npt_per_decade_integ)
        
        # Extract the spectrum: what is long is evaluating Je inside the code
        dN_dEdVdt = model.inverse_compton(energy, radius_input=radius, redshift=self._redshift).T

        return dN_dEdVdt.to('GeV-1 cm-3 s-1')


    #==================================================
    # Get the SZ rate
    #==================================================
    
    def get_rate_sz(self, frequency=np.logspace(1,3,100)*u.GHz, radius=np.logspace(0,4,100)*u.kpc,
                    Compton_only=False):
        """
        Compute the SZ emission per unit volume, or the Compton parameter per unit distance.
        
        Parameters
        ----------
        - frequency (quantity) : the physical frequency of photons
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        - Compton_only (bool): Output the Compton parameter instead of the spectrum. In the case of
        Compton only, with relativistic correction, y is weighted by the ratio f(nu,T)/f(nu,0)

        Outputs
        ----------
        - dE_dtdVdfdO (np.ndarray): the differntial production rate

        """

        # In case the input is not an array
        frequency = model_tools.check_qarray(frequency, unit='GHz')
        radius = model_tools.check_qarray(radius, unit='kpc')

        # Get the pressure and temperature profile
        radius, temperature = self.get_temperature_gas_profile(radius)
        radius, pressure    = self.get_pressure_gas_profile(radius)

        # Correct temperature for nan (e.g. beyond Rtrunc)
        temperature[np.isnan(temperature)] = 0
        
        # Get the SZ spectrum
        f_nu = cluster_szspec.tsz_spec_relativistic(frequency, temperature)    # 2D: Nfreq, Ntemp
        f_nu0 = cluster_szspec.tsz_spec_relativistic(frequency, temperature*0) # Reference non relativistic
        I0 = cluster_szspec.get_I0_CMB()

        # Get SZ power per unit volume and frequency
        if Compton_only:
            compton = model_tools.replicate_array(const.sigma_T/(const.m_e*const.c**2) * pressure,
                                                  len(frequency), T=False)
            #compton_relat = compton * f_nu/f_nu0
            output = compton.to('kpc-1')
        else:
            compton = model_tools.replicate_array(const.sigma_T/(const.m_e*const.c**2) * pressure,
                                                  len(frequency), T=False)
            dE_dtdVdfdO = compton * I0*f_nu
            output = dE_dtdVdfdO.to('eV s-1 cm-3 Hz-1 sr-1')
            
        return output

    
    #==================================================
    # Compute a Xspec table versus temperature
    #==================================================
    
    def make_xspec_table(self, Emin=0.1*u.keV, Emax=2.4*u.keV,
                         Tmin=0.1*u.keV, Tmax=50.0*u.keV, nbin=100,
                         nH=0.0/u.cm**2, file_HI=None, visu_nH=False,
                         model='APEC',
                         resp_file=None, data_file=None, app_nH_model=False):
        """
        Generate an xspec table as a function of temperature, for the cluster.
        This require xspec to be installed.
        It requires having a hydrogen column density map in the Healpix format 
        if nH should be read from a map. 
        Instrumental response files are needed for computing count rates (in ph/s)
        
        Parameters
        ----------
        - Emin (quantity): Xray band minimal energy (RASS Hard is 0.5-2.4 keV, RASS Soft is 0.1-0.4 keV)
        - Emax (quantity): Xray band maximal energy
        - Tmin (quantity): min table temperature (in keV)
        - Tmax (quantity): max table temperature (in keV)
        - nbin (int): number of temperature point in the table (in unit of cm^-2, not 10^22 cm^-2)
        - nH (quantity): H I column density at the cluster
        - file_HI (str): full path to the map file
        - visu_nH (bool): show the nH maps and histogram when extracted from data
        - model (str): which model to use (APEC or MEKAL)
        - resp_file (str): full path to the response file of e.g., ROSAT PSPC
        - data_file (str): full path to any data spectrum file needed for template in xspec 
        (see https://heasarc.gsfc.nasa.gov/FTP/rosat/doc/xselect_guide/xselect_guide_v1.1.1/xselect_ftools.pdf,
        section 5)
        - app_nH_model (bool): apply nH absorbtion to the initial model without instrumental effects

        Outputs
        ----------
        XSPEC table created in the output directory

        """
        
        # Create the output directory if needed
        if not os.path.exists(self._output_dir): os.mkdir(self._output_dir)
        
        # In case we want nH from real data at cluster location
        if file_HI is not None :
            # Make sure the FoV and resolution are ok
            if self._map_fov == None or self._map_reso == None:
                fov = 1.0
                reso = 0.1
            else:
                fov = self._map_fov.to_value('deg')
                reso = self._map_reso.to_value('deg')
                
            nH2use, nH2use_err = cluster_xspec.get_nH(file_HI,
                                                      self._coord.icrs.ra.to_value('deg'),
                                                      self._coord.icrs.dec.to_value('deg'),
                                                      fov=fov, reso=reso,
                                                      save_file=None, visu=visu_nH)
            if nH2use/nH2use_err < 5 :
                print('!!! WARNING, nH is not well constrain in the field:')
                print(('mean/rms < 5 and nH='+str(nH2use)+' 10^22 cm-2.'))

        # Or give nH directly
        elif nH is not None:
            nH2use = 10**-22 * nH.to_value('cm-2')

        # Show error
        else:
            raise ValueError("nH or file_HI should be provided to get the hydrogen column density.")
        
        # Compute the table
        cluster_xspec.make_xspec_table(self._output_dir+'/XSPEC_table.txt', nH2use,
                                       self._abundance, self._redshift,
                                       Emin.to_value('keV'), Emax.to_value('keV'), 
                                       Tmin=Tmin.to_value('keV'), Tmax=Tmax.to_value('keV'), nbin=nbin,
                                       file_ana=self._output_dir+'/xspec_analysis.txt',
                                       file_out=self._output_dir+'/xspec_analysis_output.txt',
                                       model=model,
                                       resp_file=resp_file, data_file=data_file, app_nH_model=app_nH_model,
                                       cleanup=True,
                                       logspace=True)

        # Compute the table in the case of the cluster frame
        cluster_xspec.make_xspec_table(self._output_dir+'/XSPEC_table_ClusterFrame.txt', nH2use,
                                       self._abundance, 0.0,
                                       Emin.to_value('keV'), Emax.to_value('keV'), 
                                       Tmin=Tmin.to_value('keV'), Tmax=Tmax.to_value('keV'), nbin=nbin,
                                       file_ana=self._output_dir+'/xspec_analysis.txt',
                                       file_out=self._output_dir+'/xspec_analysis_output.txt',
                                       model=model,
                                       resp_file=resp_file, data_file=data_file, app_nH_model=app_nH_model,
                                       cleanup=True,
                                       logspace=True)

    
    #==================================================
    # Read and interpolate xspec tables
    #==================================================
    
    def _itpl_xspec_table(self, xspecfile, Tinput):
        """
        Read an Xspec table and interpolate values at a given temperature

        Parameters
        ----------
        - xspecfile (str): full path to Xspec file to read

        Outputs
        ----------
        - dC (quantity): the interpolated differential counts
        - dS (quantity): the interpolated differential surface brightness counts
        - dR (quantity): the interpolated differential rate

        """
        
        file_start = 3
        
        # Read the temperature
        with open(xspecfile) as f: 
            col = list(zip(*[line.split() for line in f]))[0]
            Txspec = np.array(col[file_start:]).astype(np.float)

        # Read Xspec counts
        with open(xspecfile) as f: 
            col = list(zip(*[line.split() for line in f]))[1]
            Cxspec = np.array(col[file_start:]).astype(np.float)

        # Read Xspec surface brightness
        with open(xspecfile) as f: 
            col = list(zip(*[line.split() for line in f]))[2]
            Sxspec = np.array(col[file_start:]).astype(np.float)

        # Read Xspec rate
        with open(xspecfile) as f: 
            col = list(zip(*[line.split() for line in f]))[3]
            Rxspec = np.array(col[file_start:]).astype(np.float)
        
        # Define interpolation and set unit
        Citpl = interpolate.interp1d(Txspec, Cxspec, kind='cubic', fill_value='extrapolate')
        Sitpl = interpolate.interp1d(Txspec, Sxspec, kind='cubic', fill_value='extrapolate')
        Ritpl = interpolate.interp1d(Txspec, Rxspec, kind='cubic', fill_value='extrapolate')
        
        dCxspec = Citpl(Tinput.to_value('keV')) * 1/u.cm**2/u.s/u.cm**-5
        dSxspec = Sitpl(Tinput.to_value('keV')) * u.erg/u.cm**2/u.s/u.cm**-5
        dRxspec = Ritpl(Tinput.to_value('keV')) * 1/u.s/u.cm**-5
        
        # Take care of undefined temperature (i.e. no gas)
        dCxspec[np.isnan(Tinput.to_value('keV'))] = 0.0            
        dSxspec[np.isnan(Tinput.to_value('keV'))] = 0.0
        dRxspec[np.isnan(Tinput.to_value('keV'))] = 0.0
        if np.sum(~np.isnan(dRxspec)) == 0 :
            dRxspec[:] = np.nan
        
        return dCxspec, dSxspec, dRxspec

    
    #==================================================
    # Get the Xray rate
    #==================================================
    
    def get_rate_xray(self, radius=np.logspace(0,4,100)*u.kpc,
                      output_type='C',
                      Cframe=False):
        """
        Compute the Xray emission per unit volume. An xspec table file is needed as 
        output_dir+'/XSPEC_table.txt'. The energy band is defined in this file.
        
        Parameters
        ----------
        - radius (quantity): the physical 3d radius in units homogeneous to kpc, as a 1d array
        - output_type (str): type of output
        S == energy counts in erg/s/cm^2/sr
        C == counts in ph/s/cm^2/sr
        R == count rate in ph/s/sr (accounting for instrumental response)
        - Cframe (bool): computation assumes that we are in the cluster frame (no redshift effect)

        Outputs
        ----------
        - {dN_dtdV, dE_dtdV, dNdS_dtdV} (np.ndarray): the differential production rate

        """
        
        # In case the input is not an array
        radius = model_tools.check_qarray(radius, unit='kpc')

        # Check output type
        output_list = ['S', 'C', 'R']
        if output_type not in output_list:
            raise ValueError("Available output_type are S, C and R.")        
        
        # Get the pressure and temperature profile
        radius, T_g = self.get_temperature_gas_profile(radius)
        radius, n_e = self.get_density_gas_profile(radius)
        
        # Interpolate Xspec table at temperature values
        if Cframe:
            tab_file = self._output_dir+'/XSPEC_table_ClusterFrame.txt'
        else:
            tab_file = self._output_dir+'/XSPEC_table.txt'

        dC_xspec, dS_xspec, dR_xspec = self._itpl_xspec_table(tab_file, T_g)
        if np.sum(~np.isnan(dR_xspec)) == 0 and output_type == 'R':
            msg = ('You ask for an output in ph/s/sr (i.e. including instrumental response), ',
                   'but the xspec table was generated without response file.')
            raise ValueError(msg)

        # Get the the Xspec normalization
        mu_gas,mu_e,mu_p,mu_alpha = cluster_global.mean_molecular_weight(Y=self._helium_mass_fraction,
                                                                         Z=self._metallicity_sol*self._abundance)
        xspec_norm = 1e-14/(4*np.pi*self._D_ang**2*(1+self._redshift)**2) * n_e**2*mu_e/mu_p 

        # Compute norm * counts as dN/dtdS/dV and convert do dN/dVdt via D_lum
        if output_type == 'S':
            output = (4*np.pi*self._D_lum**2*xspec_norm * dS_xspec).to('erg s-1 cm-3')
        elif output_type == 'C':
            output = (4*np.pi*self._D_lum**2*xspec_norm * dC_xspec).to('s-1 cm-3')
        elif output_type == 'R':
            output = (4*np.pi*self._D_lum**2*xspec_norm * dR_xspec).to('s-1 cm-3 cm2')

        return output

