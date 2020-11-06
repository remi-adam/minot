"""
This file deals with 'administrative' issues regarding the Cluster Class (e.g. saving parameters etc)
"""

import os
import pprint
import numpy as np
import astropy.units as u
from astropy import constants as const
import pickle
from astropy.table import Table, Column
from astropy.io import fits

from minot.ClusterTools import map_tools

#==================================================
# Admin class
#==================================================

class Admin(object):
    """ Admin class
    This class searves as a parser to the main Cluster class, to 
    include the subclass Admin in this other file.
    
    Attributes
    ----------  
    The attributes are the same as the Cluster class, see model.py

    Methods
    ----------  
    - print_param(self): print the parameters.
    - save_param(self): save the current parameters describing the cluster object.
    - load_param(self, param_file): load a given pre-saved parameter file. The parameter
    file should contain the right parameters to avoid issues latter on.

    - save_profile(self, radius=np.logspace(0,4,1000)*u.kpc, prod_list=['all'], NR500max=5.0, 
    Npt_los=100, Energy_density=False, Epmin=None, Epmax=None, Egmin=10.0*u.MeV, Egmax=1.0*u.PeV):
    Save the profiles as fits and txt files.
    - save_spectra(self, energy=np.logspace(-2,6,1000)*u.GeV, prod_list=['all'], Rmax=None,
    NR500max=5.0, Npt_los=100): save the spectra as fits and txt files
    - save_map(self, prod_list=['all'], NR500max=5.0, Npt_los=100): save the maps as fits files
    
    - _save_txt_file(self, filename, col1, col2, col1_name, col2_name, ndec=20): internal method 
    dedicated to save data in special format

    - get_map_header(self) : return the map header.

    """
    
    #==================================================
    # Print parameters
    #==================================================
    
    def print_param(self):
        """
        Print the current parameters describing the cluster.
        
        Parameters
        ----------
        
        Outputs
        ----------
        The parameters are printed in the terminal
        
        """
        pp = pprint.PrettyPrinter(indent=4)
        
        par = self.__dict__
        keys = list(par.keys())
        
        for k in range(len(keys)):
            print(('--- '+(keys[k])[1:]))
            print(('    '+str(par[keys[k]])))
            print(('    '+str(type(par[keys[k]]))+''))

            
    #==================================================
    # Save parameters
    #==================================================
    
    def save_param(self):
        """
        Save the current parameters.
        
        Parameters
        ----------
            
        Outputs
        ----------
        The parameters are saved in the output directory

        """

        # Create the output directory if needed
        if not os.path.exists(self._output_dir): os.mkdir(self._output_dir)

        # Save
        with open(self._output_dir+'/parameters.pkl', 'wb') as pfile:
            pickle.dump(self.__dict__, pfile, pickle.HIGHEST_PROTOCOL)

        # Text file for user
        par = self.__dict__
        keys = list(par.keys())
        with open(self._output_dir+'/parameters.txt', 'w') as txtfile:
            for k in range(len(keys)):
                txtfile.write('--- '+(keys[k])[1:]+'\n')
                txtfile.write('    '+str(par[keys[k]])+'\n')
                txtfile.write('    '+str(type(par[keys[k]]))+'\n')

                
    #==================================================
    # Load parameters
    #==================================================
    
    def load_param(self, param_file):
        """
        Read the a given parameter file to re-initialize the cluster object.
        
        Parameters
        ----------
        param_file (str): the parameter file to be read
            
        Outputs
        ----------
            
        """

        with open(param_file, 'rb') as pfile:
            par = pickle.load(pfile)
            
        self.__dict__ = par

        
    #==================================================
    # Save profile
    #==================================================
    
    def save_profile(self, radius=np.logspace(0,4,100)*u.kpc,
                     Epmin=None, Epmax=None,
                     Eemin=None, Eemax=None,
                     freq0=1*u.GHz,
                     Egmin=None, Egmax=None):
        
        """
        Save the profiles in a file
        
        Parameters
        ----------
        - radius (quantity): the 3d radius in unit of kpc
        - Epmin (quantity): the lower bound for energy proton integration
        - Epmax (quantity): the upper bound for energy proton integration
        - Eemin (quantity): the lower bound for energy electron integration
        - Eemax (quantity): the upper bound for energy electron integration
        - freq0 (quantity): the frequency used to compute synchrotron emission
        - Egmin (quantity): the lower bound for energy gamma integration
        - Egmax (quantity): the upper bound for energy gamma integration

        """
        
        #========== Default keyword
        if Epmin is None:
            Epmin = self._Epmin
        if Epmax is None:
            Epmax = self._Epmax
        if Eemin is None:
            Eemin = (const.m_e*const.c**2).to('GeV')
        if Eemax is None:
            Eemax = self._Epmax
        if Egmin is None:
            Egmin = self._Epmin/10.0
        if Egmax is None:
            Egmax = self._Epmax

        #========== Create the output directory if needed
        if not os.path.exists(self._output_dir): os.mkdir(self._output_dir)
        
        #========== Create a dataframe to store all spectra in a single fits table
        tab  = Table()
        tab['Radius'] = Column(radius.to_value('kpc'), unit='kpc', description='Radius')
        
        #---------- pressure
        rad, prof = self.get_pressure_gas_profile(radius)
        tab['p_e'] = Column(prof.to_value('keV cm-3'), unit='keV cm-3', description='Thermal electron pressure')
        self._save_txt_file(self._output_dir+'/PROF_gas_pressure.txt',
                            radius.to_value('kpc'), prof.to_value('keV cm-3'), 'radius (kpc)', 'pressure (keV cm-3)')

        #---------- density
        rad, prof = self.get_density_gas_profile(radius)
        tab['n_e'] = Column(prof.to_value('cm-3'), unit='cm-3', description='Thermal electron density')
        self._save_txt_file(self._output_dir+'/PROF_gas_density.txt',
                            radius.to_value('kpc'), prof.to_value('cm-3'), 'radius (kpc)', 'density (cm-3)')

        #---------- temperature
        rad, prof = self.get_temperature_gas_profile(radius)
        tab['t_gas'] = Column(prof.to_value('keV'), unit='keV', description='Thermal gas temperature')
        self._save_txt_file(self._output_dir+'/PROF_gas_temperature.txt',
                            radius.to_value('kpc'), prof.to_value('keV'), 'radius (kpc)', 'temperature (keV)')
            
        #---------- Entropy
        rad, prof = self.get_entropy_gas_profile(radius)
        tab['k_gas'] = Column(prof.to_value('keV cm2'), unit='keV cm2', description='Thermal gas entropy')
        self._save_txt_file(self._output_dir+'/PROF_gas_entropy.txt',
                            radius.to_value('kpc'), prof.to_value('keV cm2'), 'radius (kpc)', 'entropy (keV cm2)')

        #---------- Masse HSE
        rad, prof = self.get_hse_mass_profile(radius)
        tab['m_hse'] = Column(prof.to_value('Msun'), unit='Msun', description='Enclosed hydrostatic mass')
        self._save_txt_file(self._output_dir+'/PROF_hse_mass.txt',
                            radius.to_value('kpc'), prof.to_value('Msun'), 'radius (kpc)', 'mass HSE (Msun)')

        #---------- Overdensity
        rad, prof = self.get_overdensity_contrast_profile(radius)
        tab['overdensity'] = Column(prof.to_value('adu'), unit='adu', description='Enclosed overdensity wrt critical density')
        self._save_txt_file(self._output_dir+'/PROF_overdensity.txt',
                            radius.to_value('kpc'), prof.to_value('adu'), 'radius (kpc)', 'overdensity')
        
        #---------- Gas mass
        rad, prof = self.get_gas_mass_profile(radius)
        tab['m_gas'] = Column(prof.to_value('Msun'), unit='Msun', description='Enclosed gas mass')
        self._save_txt_file(self._output_dir+'/PROF_gas_mass.txt',
                            radius.to_value('kpc'), prof.to_value('Msun'), 'radius (kpc)', 'mass gas (Msun)')

        #---------- fgas profile
        rad, prof = self.get_fgas_profile(radius)
        tab['f_gas'] = Column(prof.to_value('adu'), unit='adu', description='Enclosed gas fraction')
        self._save_txt_file(self._output_dir+'/PROF_gas_fraction.txt',
                            radius.to_value('kpc'), prof.to_value('adu'), 'radius (kpc)', 'fraction gas')

        #---------- Thermal energy
        rad, prof = self.get_thermal_energy_profile(radius)
        tab['u_th'] = Column(prof.to_value('erg'), unit='erg', description='Enclosed gas thermal energy')
        self._save_txt_file(self._output_dir+'/PROF_gas_thermal_energy.txt',
                            radius.to_value('kpc'), prof.to_value('erg'), 'radius (kpc)', 'thermal energy (erg)')

        #---------- magfield
        rad, prof = self.get_magfield_profile(radius)
        tab['B'] = Column(prof.to_value('uG'), unit='uG', description='Magnetic field')
        self._save_txt_file(self._output_dir+'/PROF_magnetic_field.txt',
                            radius.to_value('kpc'), prof.to_value('uG'), 'radius (kpc)', 'B field (uG)')
            
        #---------- Cosmic ray proton
        rad, prof = self.get_density_crp_profile(radius, Emin=Epmin, Emax=Epmax, Energy_density=False)
        tab['n_crp'] = Column(prof.to_value('cm-3'), unit='cm-3', description='Cosmic ray proton density')
        self._save_txt_file(self._output_dir+'/PROF_crp_density.txt',
                            radius.to_value('kpc'), prof.to_value('cm-3'), 'radius (kpc)', 'density (cm-3)')

        #---------- Cosmic ray to thermal energy
        rad, prof = self.get_crp_to_thermal_energy_profile(radius, Emin=Epmin, Emax=Epmax)
        tab['x_crp'] = Column(prof.to_value('adu'), unit='adu', description='Enclosed cosmic ray to thermal energy')
        self._save_txt_file(self._output_dir+'/PROF_fraction_energy_cosmic_to_thermal.txt',
                            radius.to_value('kpc'), prof.to_value('adu'),
                            'radius (kpc)', 'x')

        #---------- Cosmic ray electron secondaries
        rad, prof = self.get_density_cre2_profile(radius, Emin=Eemin, Emax=Eemax, Energy_density=False)
        tab['n_cre'] = Column(prof.to_value('cm-3'), unit='cm-3', description='Cosmic ray electron density')
        self._save_txt_file(self._output_dir+'/PROF_cre_density.txt',
                            radius.to_value('kpc'), prof.to_value('cm-3'), 'radius (kpc)', 'density (cm-3)')

        #---------- Gamma surface brightness
        rad, prof = self.get_gamma_profile(radius, Emin=Egmin, Emax=Egmax, Energy_density=False)
        tab['Sg'] = Column(prof.to_value('cm-2 s-1 sr-1'), unit='cm-2 s-1 sr-1', description='Gamma surface brightness')
        self._save_txt_file(self._output_dir+'/PROF_gamma_surface_brightness.txt',
                            radius.to_value('kpc'), prof.to_value('cm-2 s-1 sr-1'), 'radius (kpc)',
                            'gamma SB (cm-2 s-1 sr-1)')
        
        #---------- Gamma integrated flux
        prof = self.get_gamma_flux(Rmax=radius, Emin=Egmin, Emax=Egmax, Energy_density=False, type_integral='spherical')
        tab['Fg'] = Column(prof.to_value('cm-2 s-1'), unit='cm-2 s-1', description='Gamma flux')
        self._save_txt_file(self._output_dir+'/PROF_gamma_flux.txt',
                            radius.to_value('kpc'), prof.to_value('cm-2 s-1'), 'radius (kpc)', 'gamma flux (cm-2 s-1)')

        #---------- Neutrinos surface brightness
        rad, prof = self.get_neutrino_profile(radius, Emin=Egmin, Emax=Egmax, Energy_density=False, flavor='all')
        tab['Snu'] = Column(prof.to_value('cm-2 s-1 sr-1'), unit='cm-2 s-1 sr-1', description='Neutrino surface brightness')
        self._save_txt_file(self._output_dir+'/PROF_neutrino_surface_brightness.txt',
                            radius.to_value('kpc'), prof.to_value('cm-2 s-1 sr-1'), 'radius (kpc)',
                            'neutrino SB (cm-2 s-1 sr-1)')
        
        #---------- Neutrinos integrated flux
        prof = self.get_neutrino_flux(Rmax=radius, Emin=Egmin, Emax=Egmax, Energy_density=False, type_integral='spherical',
                                      flavor='all')
        tab['Fnu'] = Column(prof.to_value('cm-2 s-1'), unit='cm-2 s-1', description='Neutrino flux')
        self._save_txt_file(self._output_dir+'/PROF_neutrino_flux.txt',
                            radius.to_value('kpc'), prof.to_value('cm-2 s-1'), 'radius (kpc)', 'neutrino flux (cm-2 s-1)')

        #---------- IC surface brightness
        rad, prof = self.get_ic_profile(radius, Emin=Egmin, Emax=Egmax, Energy_density=False)
        tab['Sic'] = Column(prof.to_value('cm-2 s-1 sr-1'), unit='cm-2 s-1 sr-1', description='IC surface brightness')
        self._save_txt_file(self._output_dir+'/PROF_ic_surface_brightness.txt',
                            radius.to_value('kpc'), prof.to_value('cm-2 s-1 sr-1'), 'radius (kpc)',
                            'IC SB (cm-2 s-1 sr-1)')
        
        #---------- IC integrated flux
        prof = self.get_ic_flux(Rmax=radius, Emin=Egmin, Emax=Egmax, Energy_density=False, type_integral='spherical')
        tab['Fic'] = Column(prof.to_value('cm-2 s-1'), unit='cm-2 s-1', description='IC flux')
        self._save_txt_file(self._output_dir+'/PROF_ic_flux.txt',
                            radius.to_value('kpc'), prof.to_value('cm-2 s-1'), 'radius (kpc)', 'IC flux (cm-2 s-1)')
        
        #---------- Synchrotron surface brightness
        rad, prof = self.get_synchrotron_profile(radius, freq0=freq0)
        tab['Ssynch'] = Column(prof.to_value('Jy sr-1'), unit='Jy sr-1', description='Synchrotron surface brightness')
        self._save_txt_file(self._output_dir+'/PROF_synchrotron_surface_brightness.txt',
                            radius.to_value('kpc'), prof.to_value('Jy sr-1'), 'radius (kpc)', 'Synch SB (Jy sr-1)')
        
        #---------- Synchrotron integrated flux
        prof = self.get_synchrotron_flux(Rmax=radius, freq0=freq0, type_integral='spherical')
        tab['Fsynch'] = Column(prof.to_value('Jy'), unit='Jy', description='Synchrotron flux')
        self._save_txt_file(self._output_dir+'/PROF_synchrotron_flux.txt',
                            radius.to_value('kpc'), prof.to_value('Jy'), 'radius (kpc)', 'Synch flux (Jy)')
        
        #---------- Compton parameter profile
        rad, prof = self.get_sz_profile(radius, Compton_only=True)
        tab['Ssz'] = Column(prof.to_value('adu'), unit='adu', description='Compton parameter')
        self._save_txt_file(self._output_dir+'/PROF_sz_compton.txt',
                            radius.to_value('kpc'), prof.to_value('adu'), 'radius (kpc)', 'SZ Compton (adu)')
        
        #---------- Synchrotron integrated flux
        prof = self.get_sz_flux(Rmax=radius, Compton_only=True, type_integral='spherical')
        tab['Fsz'] = Column(prof.to_value('kpc2'), unit='kpc2', description='SZ flux')
        self._save_txt_file(self._output_dir+'/PROF_sz_flux.txt',
                            radius.to_value('kpc'), prof.to_value('kpc2'), 'radius (kpc)', 'SZ flux (kpc2)')
        
        #++++++++++ X-ray needs tabulated XSPEC
        if os.path.exists(self._output_dir+'/XSPEC_table.txt'):
            #---------- Xray profile
            rad, prof = self.get_xray_profile(radius, output_type='C')
            tab['Sx'] = Column(prof.to_value('s-1 cm-2 sr-1'), unit='s-1 cm-2 sr-1', description='Xray surface brightness')
            self._save_txt_file(self._output_dir+'/PROF_x_surface_brightness.txt',
                                radius.to_value('kpc'), prof.to_value('s-1 cm-2 sr-1'), 'radius (kpc)', 'X SB (s-1 cm-2 sr-1)')
            
            #---------- Xray integrated flux
            prof = self.get_xray_flux(Rmax=radius, output_type='C', type_integral='spherical')
            tab['Fx'] = Column(prof.to_value('s-1 cm-2'), unit='s-1 cm-2', description='X flux')
            self._save_txt_file(self._output_dir+'/PROF_x_flux.txt',
                                radius.to_value('kpc'), prof.to_value('s-1 cm-2'), 'radius (kpc)', 'X flux (s-1 cm-2)')
        else:
            if not self._silent:
                print('!!! WARNING: XSPEC_table.txt not generated, skip Xray observables')


        #========== Save the data frame in a single file as well
        tab.meta['comments'] = ['Proton spectra are integrated within '+str(Epmin)+' and '+str(Epmax)+'.',
                                'Electron spectra are integrated within '+str(Eemin)+' and '+str(Eemax)+'.',
                                'Gamma ray spectra are integrated within '+str(Egmin)+' and '+str(Egmax)+'.',
                                'Neutrino ray spectra are integrated within '+str(Egmin)+' and '+str(Egmax)+'.',
                                'Inverse Compton spectra are integrated within '+str(Egmin)+' and '+str(Egmax)+'.',
                                'Synchrotron emission is computed at '+str(freq0)+'.']

        tab.write(self._output_dir+'/PROFILE.fits', overwrite=True)
        
        
    #==================================================
    # Save spectra
    #==================================================
    
    def save_spectra(self, energy=np.logspace(-2,7,100)*u.GeV,
                     energyX=np.linspace(0.1,20,100)*u.keV,
                     frequency=np.logspace(-2,3,100)*u.GHz,
                     Egmax=None,
                     Rmax=None):
        """
        Save the spectra
        
        Parameters
        ----------
        - energy (quantity) : the physical energy for CR related quantities
        - frequency (quantity) : the frequency of synchrotron and SZ
        - energyX (quantity) : the physical energy for Xray
        - Rmax (quantity): the radius within with the spectrum is computed 
        (default is R500)
        - Egmax (quantity): the upper bound for energy gamma integration

        Outputs
        ----------
        Files are saved

        """

        if Rmax == None:
            Rmax = self._R500
        if Egmax is None:
            Egmax = self._Epmax

        # Create the output directory if needed
        if not os.path.exists(self._output_dir): os.mkdir(self._output_dir)

        # Create a dataframe to store all spectra in a single fits table
        tab1  = Table()
        tab1['Energy'] = Column(energy.to_value('MeV'), unit='MeV', description='Energy')

        tab2  = Table()
        tab2['Frequency'] = Column(frequency.to_value('GHz'), unit='GHz', description='Frequency')

        tab3  = Table()
        tab3['Energy'] = Column(energyX.to_value('keV'), unit='keV', description='Xray energy')
        
        #---------- proton spectrum
        eng, spec = self.get_crp_spectrum(energy, Rmax=Rmax)
        tab1['CRp'] = Column(spec.to_value('MeV-1'), unit='MeV-1', description='Cosmic ray proton spectrum')
        self._save_txt_file(self._output_dir+'/SPECTRA_cosmic_ray_proton.txt',
                            eng.to_value('MeV'), spec.to_value('MeV-1'), 'energy (MeV)', 'spectrum (MeV-1)')
        
        #---------- secondary electron spectrum
        eng, spec = self.get_cre2_spectrum(energy, Rmax=Rmax)
        tab1['CRe'] = Column(spec.to_value('MeV-1'), unit='MeV-1', description='Cosmic ray electron spectrum')
        self._save_txt_file(self._output_dir+'/SPECTRA_cosmic_ray_electron.txt',
                            eng.to_value('MeV'), spec.to_value('MeV-1'), 'energy (MeV)', 'spectrum (MeV-1)')
        
        #---------- gamma spectrum
        eng, spec = self.get_gamma_spectrum(energy, Rmax=Rmax, type_integral='spherical')
        tab1['gamma'] = Column(spec.to_value('MeV-1 cm-2 s-1'), unit='MeV-1 cm-2 s-1', description='Gamma ray spectrum')
        self._save_txt_file(self._output_dir+'/SPECTRA_gamma.txt',
                            eng.to_value('MeV'), spec.to_value('MeV-1 cm-2 s-1'), 'energy (MeV)', 'spectrum (MeV-1 cm-2 s-1)')
        
        #---------- gamma flux spectrum
        spec = self.get_gamma_flux(Emin=energy, Emax=Egmax, Rmax=Rmax, Energy_density=False, type_integral='spherical')
        tab1['gammaF'] = Column(spec.to_value('cm-2 s-1'), unit='cm-2 s-1', description='Integrated gamma ray spectrum')
        self._save_txt_file(self._output_dir+'/SPECTRA_gammaF.txt',
                            eng.to_value('MeV'), spec.to_value('cm-2 s-1'), 'energy (MeV)', 'spectrum (cm-2 s-1)')
        
        #---------- neutrino spectrum
        eng, spec = self.get_neutrino_spectrum(energy, Rmax=Rmax, type_integral='spherical', flavor='all')
        tab1['nu'] = Column(spec.to_value('MeV-1 cm-2 s-1'), unit='MeV-1 cm-2 s-1', description='Neutrino spectrum')
        self._save_txt_file(self._output_dir+'/SPECTRA_neutrino.txt',
                            eng.to_value('MeV'), spec.to_value('MeV-1 cm-2 s-1'), 'energy (MeV)', 'spectrum (MeV-1 cm-2 s-1)')
        
        #---------- neutrino flux spectrum
        spec = self.get_neutrino_flux(Emin=energy,Emax=Egmax,Rmax=Rmax,Energy_density=False,type_integral='spherical',flavor='all')
        tab1['nuF'] = Column(spec.to_value('cm-2 s-1'), unit='cm-2 s-1', description='Integrated neutrino spectrum')
        self._save_txt_file(self._output_dir+'/SPECTRA_neutrinoF.txt',
                            eng.to_value('MeV'), spec.to_value('cm-2 s-1'), 'energy (MeV)', 'spectrum (cm-2 s-1)')
        
        #---------- IC spectrum
        eng, spec = self.get_ic_spectrum(energy, Rmax=Rmax, type_integral='spherical')
        tab1['IC'] = Column(spec.to_value('MeV-1 cm-2 s-1'), unit='MeV-1 cm-2 s-1', description='IC spectrum')
        self._save_txt_file(self._output_dir+'/SPECTRA_IC.txt',
                            eng.to_value('MeV'), spec.to_value('MeV-1 cm-2 s-1'), 'energy (MeV)', 'spectrum (MeV-1 cm-2 s-1)')
        
        #---------- IC flux spectrum
        spec = self.get_ic_flux(Emin=energy, Emax=Egmax, Rmax=Rmax, Energy_density=False, type_integral='spherical')
        tab1['ICF'] = Column(spec.to_value('cm-2 s-1'), unit='cm-2 s-1', description='Integrated IC spectrum')
        self._save_txt_file(self._output_dir+'/SPECTRA_ICF.txt',
                            eng.to_value('MeV'), spec.to_value('cm-2 s-1'), 'energy (MeV)', 'spectrum (cm-2 s-1)')
        
        #---------- Synchrotron spectrum
        freq, spec = self.get_synchrotron_spectrum(frequency, Rmax=Rmax, type_integral='spherical')
        tab2['synch'] = Column(spec.to_value('Jy'), unit='Jy', description='Synch spectrum')
        self._save_txt_file(self._output_dir+'/SPECTRA_Synch.txt',
                            freq.to_value('GHz'), spec.to_value('Jy'), 'frequency (GHz)', 'spectrum (Jy)')
        
        #---------- SZ spectrum
        freq, spec = self.get_sz_spectrum(frequency, Rmax=Rmax, type_integral='spherical', Compton_only=False)
        tab2['SZ'] = Column(spec.to_value('Jy'), unit='Jy', description='SZ spectrum')
        self._save_txt_file(self._output_dir+'/SPECTRA_SZ.txt',
                            freq.to_value('GHz'), spec.to_value('Jy'), 'frequency (GHz)', 'spectrum (Jy)')

        #---------- Xray spectrum
        if os.path.exists(self._output_dir+'/XSPEC_table.txt'):

            engX, spec = self.get_xray_spectrum(energyX, Rmax=Rmax, type_integral='spherical', output_type='C')
            tab3['Xray'] = Column(spec.to_value('s-1 cm-2 keV-1'), unit='s-1 cm-2 keV-1', description='Xray spectrum')
            self._save_txt_file(self._output_dir+'/SPECTRA_Xray.txt',
                                engX.to_value('keV'), spec.to_value('s-1 cm-2 keV-1'), 'energy (keV)', 'spectrum (s-1 cm-2 keV-1)')
        else:
            print('!!! WARNING: XSPEC_table.txt not generated, skip Xray observables')
            
        #========== Save the data frame in a single file as well
        tab1.meta['comments'] = ['Spectra are computed within '+str(Rmax.to_value('kpc'))+' kpc.']
        tab2.meta['comments'] = ['Spectra are computed within '+str(Rmax.to_value('kpc'))+' kpc.']
        tab3.meta['comments'] = ['Spectra are computed within '+str(Rmax.to_value('kpc'))+' kpc.']
        tab1.write(self._output_dir+'/SPECTRA1.fits', overwrite=True)
        tab2.write(self._output_dir+'/SPECTRA2.fits', overwrite=True)
        tab3.write(self._output_dir+'/SPECTRA3.fits', overwrite=True)

        # Merging
        hdul = fits.open(self._output_dir+'/SPECTRA1.fits')
        hdul2 = fits.open(self._output_dir+'/SPECTRA2.fits')
        hdul3 = fits.open(self._output_dir+'/SPECTRA3.fits')
        hdul.append(hdul2[1])
        hdul.append(hdul3[1])
        hdul.writeto(self._output_dir+'/SPECTRA.fits', overwrite=True)

        #Cleaning
        os.remove(self._output_dir+'/SPECTRA1.fits')
        os.remove(self._output_dir+'/SPECTRA2.fits')
        os.remove(self._output_dir+'/SPECTRA3.fits')
        
    #==================================================
    # Save map
    #==================================================
    
    def save_map(self, Normalize=False,
                 freq0=1*u.GHz,
                 Egmin=None, Egmax=None):
        """
        Save the maps in a file
        
        Parameters
        ----------
        - Normalize (bool): to normalize the map to the total flux
        - freq0 (quantity): the frequency used to compute synchrotron emission
        - Egmin (quantity): the lower bound for energy gamma integration
        - Egmax (quantity): the upper bound for energy gamma integration

        Outputs
        ----------
        Files are saved

        """
        
        #========== Default keyword
        if Egmin is None:
            Egmin = self._Epmin/10.0
        if Egmax is None:
            Egmax = self._Epmax

        #========== Create the output directory if needed
        if not os.path.exists(self._output_dir): os.mkdir(self._output_dir)

        #========== Get the header of the maps
        header = self.get_map_header()

        #---------- gamma    
        image = self.get_gamma_map(Emin=Egmin, Emax=Egmax, Energy_density=False, Normalize=Normalize)
        hdu = fits.PrimaryHDU(header=header)
        hdu.data = image.value
        hdu.header.add_comment('Gamma map')
        hdu.header.add_comment('Unit = '+str(image.unit))
        hdu.writeto(self._output_dir+'/MAP_gamma.fits', overwrite=True)

        #---------- neutrino    
        image = self.get_neutrino_map(Emin=Egmin, Emax=Egmax, Energy_density=False, Normalize=Normalize, flavor='all')
        hdu = fits.PrimaryHDU(header=header)
        hdu.data = image.value
        hdu.header.add_comment('Neutrino map')
        hdu.header.add_comment('Unit = '+str(image.unit))
        hdu.writeto(self._output_dir+'/MAP_neutrino.fits', overwrite=True)

        #---------- IC    
        image = self.get_ic_map(Emin=Egmin, Emax=Egmax, Energy_density=False, Normalize=Normalize)
        hdu = fits.PrimaryHDU(header=header)
        hdu.data = image.value
        hdu.header.add_comment('Inverse Compton map')
        hdu.header.add_comment('Unit = '+str(image.unit))
        hdu.writeto(self._output_dir+'/MAP_sz.fits', overwrite=True)

        #---------- Synchrotron
        image = self.get_synchrotron_map(freq0=freq0, Normalize=Normalize)
        hdu = fits.PrimaryHDU(header=header)
        hdu.data = image.value
        hdu.header.add_comment('Synchrotron map')
        hdu.header.add_comment('Unit = '+str(image.unit))
        hdu.writeto(self._output_dir+'/MAP_synchrotron.fits', overwrite=True)

        #---------- SZ ymap
        image = self.get_sz_map(Compton_only=True, Normalize=Normalize)
        hdu = fits.PrimaryHDU(header=header)
        hdu.data = image.value
        hdu.header.add_comment('SZ Compton map')
        hdu.header.add_comment('Unit = '+str(image.unit))
        hdu.writeto(self._output_dir+'/MAP_sz.fits', overwrite=True)

        #---------- Xray
        if os.path.exists(self._output_dir+'/XSPEC_table.txt'):
            image = self.get_xray_map(output_type='C', Normalize=Normalize)
            hdu = fits.PrimaryHDU(header=header)
            hdu.data = image.value
            hdu.header.add_comment('Xray map')
            hdu.header.add_comment('Unit = '+str(image.unit))
            hdu.writeto(self._output_dir+'/MAP_xray.fits', overwrite=True)

        else:
            if not self._silent:
                print('!!! WARNING: XSPEC_table.txt not generated, skip Xray observables')
    
    
    #==================================================
    # Saving txt file utility function
    #==================================================
    
    def _save_txt_file(self, filename, col1, col2, col1_name, col2_name, ndec=20):
        """
        Save the file with a given format in txt file
        
        Parameters
        ----------
        - filename (str): full path to the file
        - col1 (np.ndarray): the first column of data
        - col2 (np.ndarray): the second column of data
        - col1_name (str): the name of the first column
        - col2_name (str): the name of the second column
        - ndec (int): number of decimal in numbers
        
        Outputs
        ----------
        Files are saved

        """
        
        ncar = ndec + 6

        # Mae sure name are not too long
        col1_name = ('{:.'+str(ncar-1)+'}').format(col1_name)
        col2_name = ('{:.'+str(ncar)+'}').format(col2_name)

        # Correct formating
        col1_name = ('{:>'+str(ncar-1)+'}').format(col1_name)
        col2_name = ('{:>'+str(ncar)+'}').format(col2_name)

        # saving
        sfile = open(filename, 'w')
        sfile.writelines(['#'+col1_name, ('{:>'+str(ncar)+'}').format(''), col2_name+'\n'])
        for il in range(len(col1)):
            sfile.writelines([('{:.'+str(ndec)+'e}').format(col1[il]),
                              ('{:>'+str(ncar)+'}').format(''),
                              ('{:.'+str(ndec)+'e}').format(col2[il])+'\n'])
        sfile.close()
        
        
    #==================================================
    # Extract the header
    #==================================================
    
    def get_map_header(self):
        """
        Extract the header of the map
        
        Parameters
        ----------

        Outputs
        ----------
        - header (astropy object): the header associated to the map

        """

        # Get the needed parameters in case of map header
        if self._map_header is not None:
            header = self._map_header
            
        # Get the needed parameters in case of set-by-hand map parameters
        elif (self._map_coord is not None) and (self._map_reso is not None) and (self._map_fov is not None):
            header = map_tools.define_std_header(self._map_coord.icrs.ra.to_value('deg'),
                                                 self._map_coord.icrs.dec.to_value('deg'),
                                                 self._map_fov.to_value('deg')[0],
                                                 self._map_fov.to_value('deg')[1],
                                                 self._map_reso.to_value('deg'))
            
        # Otherwise there is a problem
        else:
            raise TypeError("A header, or the map_coord & map_reso & map_fov should be defined.")

        return header        
    
    
