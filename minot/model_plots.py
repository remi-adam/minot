"""
This file contain a subclass of the model.py module and Cluster class. It
is dedicated to the computing of observables.

"""

#==================================================
# Requested imports
#==================================================

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import SymLogNorm
import astropy.units as u
import numpy as np
from astropy.wcs import WCS
from astropy import constants as const
import os

#==================================================
# Style
#==================================================

cta_energy_range   = [0.02, 100.0]*u.TeV
fermi_energy_range = [0.1, 300.0]*u.GeV

def set_default_plot_param():
    
    dict_base = {'font.size':        16, 
                 'legend.fontsize':  16,
                 'xtick.labelsize':  16,
                 'ytick.labelsize':  16,
                 'axes.labelsize':   16,
                 'axes.titlesize':   16,
                 'figure.titlesize': 16,
                 'figure.figsize':[8.0, 6.0],
                 'figure.subplot.right':0.97,
                 'figure.subplot.left':0.18, # Ensure enough space on the left so that all plot can be aligned
                 'font.family':'serif',
                 'figure.facecolor': 'white',
                 'legend.frameon': True}

    plt.rcParams.update(dict_base)

#==================================================
# Plot radial profiles
#==================================================

def profile(radius, angle, prof, filename, label='Profile', R500=None):
    """
    Plot the profiles
    
    Parameters
    ----------
    - radius (quantity): homogeneous to kpc
    - angle (quantity): homogeneous to deg
    - prof (quantity): any profile
    - label (str): the full name of the profile
    - filename (str): the full path name of the profile
    - R500 (quantity): homogeneous to kpc

    """

    p_unit = prof.unit
    r_unit = radius.unit
    t_unit = angle.unit

    wgood = ~np.isnan(prof)
    profgood = prof[wgood]
    ymin = np.nanmin(profgood[profgood>0].to_value())*0.5
    ymax = np.nanmax(profgood[profgood>0].to_value())*2.0
    
    fig, ax1 = plt.subplots()
    ax1.plot(radius, prof, 'blue')
    if R500 is not None:
        ax1.axvline(R500.to_value(r_unit), ymin=-1e300, ymax=1e300,
                    color='black', label='$R_{500}$', linestyle='--')
    ax1.set_xlabel('Radius ('+str(r_unit)+')')
    ax1.set_ylabel(label)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim([np.amin(radius.to_value()), np.amax(radius.to_value())])
    ax1.set_ylim([ymin,ymax])
    ax1.legend()
    
    # Add extra projected radius axis
    ax2 = ax1.twiny()
    ax2.plot(angle, prof, 'blue')
    ax2.set_xlabel('Radius ('+str(t_unit)+')', color='k')
    ax2.set_xscale('log')
    ax2.set_xlim([np.amin(angle.to_value()),np.amax(angle.to_value())])
    fig.savefig(filename)
    plt.close()

    
#==================================================
# Plot spectra
#==================================================

def spectra(energy, freq, spec, filename, label='Spectrum'):
    """
    Plot the profiles
    
    Parameters
    ----------
    - energy (quantity): homogeneous to GeV
    - sepc (quantity): any spectrum
    - sepc_label (str): the full name of the sepctrum
    - filename (str): the full path name of the profile

    """

    s_unit = spec.unit
    e_unit = energy.unit
    f_unit = freq.unit

    wgood = ~np.isnan(spec)
    specgood = spec[wgood]
    ymin = np.nanmin(specgood[specgood>0].to_value())*0.5
    ymax = np.nanmax(specgood[specgood>0].to_value())*2.0
        
    fig, ax = plt.subplots()
    ax.plot(energy, spec, 'black')
    ax.fill_between(cta_energy_range.to_value(e_unit), ymin, ymax,
                    facecolor='blue', alpha=0.2, label='CTA range')
    ax.fill_between(fermi_energy_range.to_value(e_unit), ymin, ymax,
                    facecolor='red', alpha=0.2, label='Fermi range')
    ax.set_xlabel('Energy ('+str(e_unit)+')')
    ax.set_ylabel(label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim([ymin,ymax])
    ax.set_xlim([np.amin(energy.to_value()), np.amax(energy.to_value())])
    plt.legend()

    # Add extra projected frequency axis
    ax2 = ax.twiny()
    ax2.plot(freq, spec, 'black')
    ax2.set_xlabel('Frequency ('+str(f_unit)+')', color='k')
    ax2.set_xscale('log')
    ax2.set_xlim([np.amin(freq.to_value()),np.amax(freq.to_value())])
    
    fig.savefig(filename)
    plt.close()


#==================================================
# Plot spectra
#==================================================

def maps(image, header, filename,
         label='Map', coord=None, theta_500=None,
         theta_trunc=None, logscale=False):
    """
    Plot the profiles
    
    Parameters
    ----------
    - image (np.2darray): the map
    - header (str): corresponding header
    - filename (str): the full path name of the profile
    - theta_500 (quantity): angle corresponding to R500
    - theta_trunc (quantity): angle corresponding to the truncation

    """

    plt.rcParams.update({'figure.subplot.right':0.90,
                         'figure.subplot.left':0.05})

    wcs_map = WCS(header)

    #---------- Check the map scale
    if np.amin(image) == np.amax(image):
        logscale = False
        print('WARNING: the image is empty.')
        print('         You may have set the map coordinates far away from the cluster center.')


    #---------- Get vmin/max
    vmax = np.nanmax(image)
    vmin = vmax/1e4
        
    #---------- Plot the map
    fig = plt.figure()
    ax = plt.subplot(projection=wcs_map)
    if logscale:
        try: # base argument should be there in python 3 but not in python 2
            plt.imshow(image, origin='lower', cmap='magma', norm=SymLogNorm(vmin, vmin=vmin, vmax=vmax, base=10))
        except:
            plt.imshow(image, origin='lower', cmap='magma', norm=SymLogNorm(vmin, vmin=vmin, vmax=vmax))
    else:
        plt.imshow(image, origin='lower', cmap='magma')
        
    if coord is not None and theta_500 is not None:
        circle = Ellipse((coord.icrs.ra.deg, coord.icrs.dec.deg),
                         2*theta_500.to_value('deg')/np.cos(coord.icrs.dec.rad),
                         2*theta_500.to_value('deg'),
                         linewidth=2, fill=False, zorder=2,
                         edgecolor='white', linestyle='-.',
                         facecolor='none', transform=ax.get_transform('fk5'))
        ax.add_patch(circle)
        txt = plt.text(coord.icrs.ra.deg - theta_500.to_value('deg'),
                       coord.icrs.dec.deg - theta_500.to_value('deg'),
                       '$R_{500}$',
                       transform=ax.get_transform('fk5'), fontsize=10, color='white',
                       horizontalalignment='center',verticalalignment='center')
        
    if coord is not None and theta_trunc is not None:
        circle = Ellipse((coord.icrs.ra.deg, coord.icrs.dec.deg),
                         2*theta_trunc.to_value('deg')/np.cos(coord.icrs.dec.rad),
                         2*theta_trunc.to_value('deg'),
                         linewidth=2, fill=False, zorder=2,
                         edgecolor='white', linestyle='--',
                         facecolor='none', transform=ax.get_transform('fk5'))
        ax.add_patch(circle)
        txt = plt.text(coord.icrs.ra.deg - theta_trunc.to_value('deg'),
                       coord.icrs.dec.deg - theta_trunc.to_value('deg'),
                       '$R_{trunc}$',
                       transform=ax.get_transform('fk5'), fontsize=10, color='white',
                       horizontalalignment='center',verticalalignment='center')
        
    ax.set_xlabel('R.A. (deg)')
    ax.set_ylabel('Dec. (deg)')
    cbar = plt.colorbar()
    cbar.set_label(label)
    fig.savefig(filename)
    plt.close()
    set_default_plot_param()

#==================================================
# Main function
#==================================================

class Plots(object):
    """ Observable class
    This class serves as a parser to the main Cluster class, to 
    include the subclass Observable in this other file.

    Attributes
    ----------  
    The attributes are the same as the Cluster class, see model.py

    Methods
    ----------  
    - plot(self, list_prod=['all'],radius=np.logspace(0,4,1000)*u.kpc, 
    energy=np.logspace(-2,6,1000)*u.GeV, NR500max=5.0, Npt_los=100, Rmax=None,
    Epmin=None, Epmax=None, Egmin=10.0*u.MeV, Egmax=1.0*u.PeV): 

    """

    #==================================================
    # Main plot function
    #==================================================

    def plot(self, prod_list=['all'],
             radius=np.logspace(0,4,100)*u.kpc,
             energy=np.logspace(-2,7,100)*u.GeV,
             energyX=np.linspace(0.1,20,100)*u.keV,
             frequency=np.logspace(-2,3,100)*u.GHz,
             Epmin=None, Epmax=None,
             Eemin=None, Eemax=None,
             freq0=1*u.GHz,
             Rmax=None,
             Egmin=None, Egmax=None,
             directory=None):
        
        """
        Main function of the sub-module of the cluster class dedicated to plots.
        
        Parameters
        ----------
        - prod_list (list): the list of what is required for production
        - radius (quantity) : the physical radius
        - energy (quantity) : the physical energy of CR protons
        - NR500max (float): the integration will stop at NR500max x R500
        Only used for projected profiles.
        - Npt_los (int): the number of points for line of sight integration
        Only used for projected profiles.
        - Epmin (quantity): the lower bound for energy proton integration
        - Epmax (quantity): the upper bound for energy proton integration
        - Egmin (quantity): the lower bound for energy gamma integration
        - Egmax (quantity): the upper bound for energy gamma integration
        - Rmax (quantity): the radius within with the spectrum is computed 
        (default is R500)
        - directory (str): full path to a directory where to produce plots. 
        Default will be self.output_dir
        
        """

        # Default keyword
        if Epmin is None:
            Epmin = self._Epmin
        if Epmax is None:
            Epmax = self._Epmax
        if Eemin is None:
            Eemin = (const.m_e*const.c**2).to('GeV')
        if Eemax is None:
            Eemax = self._Epmax
        if Rmax  is None:
            Rmax = self._R500
        if Egmin is None:
            Egmin = self._Epmin/10.0
        if Egmax is None:
            Egmax = self._Epmax

        # Directory
        if directory is not None:
            mydir = directory
        else:
            mydir = self._output_dir
        if not os.path.exists(mydir): os.mkdir(mydir)

        # plot parameters
        set_default_plot_param()

        Egstrlim = str(round(Egmin.value,2)*Egmin.unit)+'-'+str(round(Egmax.value,2)*Egmax.unit)

        #---------- Profiles
        if 'all' in prod_list or 'profile' in prod_list:
            angle = (radius.to_value('kpc')/self._D_ang.to_value('kpc')*180.0/np.pi)*u.deg

            # Pressure
            rad, prof = self.get_pressure_gas_profile(radius)
            profile(radius, angle, prof.to('keV cm-3'), mydir+'/PLOT_PROF_gas_pressure.pdf',
                    label='Electron pressure (keV cm$^{-3}$)', R500=self._R500)
            if not self._silent: print('----- Plot done: gas pressure')

            # Density
            rad, prof = self.get_density_gas_profile(radius)
            profile(radius, angle, prof.to('cm-3'), mydir+'/PLOT_PROF_gas_density.pdf',
                    label='Electron density (cm$^{-3}$)', R500=self._R500)
            if not self._silent: print('----- Plot done: gas density')

            # temperature
            rad, prof = self.get_temperature_gas_profile(radius)
            profile(radius, angle, prof.to('keV'), mydir+'/PLOT_PROF_gas_temperature.pdf',
                    label='Gas temperature (keV)', R500=self._R500)
            if not self._silent: print('----- Plot done: gas temperature')

            # Entropy
            rad, prof = self.get_entropy_gas_profile(radius)
            profile(radius, angle, prof.to('keV cm2'), mydir+'/PLOT_PROF_gas_entropy.pdf',
                    label='Gas entropy (keV cm$^2$)', R500=self._R500)
            if not self._silent: print('----- Plot done: gas entropy')

            # Masse HSE
            rad, prof = self.get_hse_mass_profile(radius)
            profile(radius, angle, prof.to('Msun'), mydir+'/PLOT_PROF_hse_mass.pdf',
                    label='HSE mass (M$_{\\odot}$)', R500=self._R500)
            if not self._silent: print('----- Plot done: HSE mass')

            # Overdensity
            rad, prof = self.get_overdensity_contrast_profile(radius)
            profile(radius, angle, prof.to('adu'), mydir+'/PLOT_PROF_overdensity.pdf',
                    label='Overdensity $\\rho / \\rho_{c}$', R500=self._R500)
            if not self._silent: print('----- Plot done: density contrast')

            # Gas mass
            rad, prof = self.get_gas_mass_profile(radius)
            profile(radius, angle, prof.to('Msun'), mydir+'/PLOT_PROF_gas_mass.pdf',
                    label='Gas mass (M$_{\\odot}$)', R500=self._R500)
            if not self._silent: print('----- Plot done: gas mass')

            # fgas profile
            rad, prof = self.get_fgas_profile(radius)
            profile(radius, angle, prof.to('adu'), mydir+'/PLOT_PROF_gas_fraction.pdf',
                    label='Gas fraction', R500=self._R500)
            if not self._silent: print('----- Plot done: gas fraction')

            # Thermal energy
            rad, prof = self.get_thermal_energy_profile(radius)
            profile(radius, angle, prof.to('erg'), mydir+'/PLOT_PROF_gas_thermal_energy.pdf',
                    label='Thermal energy (erg)', R500=self._R500)
            if not self._silent: print('----- Plot done: thermal energy')

            # Magfield
            rad, prof = self.get_magfield_profile(radius)
            profile(radius, angle, prof.to('uG'), mydir+'/PLOT_PROF_magnetic_field.pdf',
                    label='Magnetic field ($\\mu$G)', R500=self._R500)
            if not self._silent: print('----- Plot done: magnetic field')
            
            # Cosmic ray proton
            if self._X_crp_E['X'] > 0:
                rad, prof = self.get_density_crp_profile(radius, Emin=Epmin, Emax=Epmax, Energy_density=False)
                profile(radius, angle, prof.to('cm-3'), mydir+'/PLOT_PROF_crp_density.pdf',
                        label='CRp density (cm$^{-3}$)', R500=self._R500)
                if not self._silent: print('----- Plot done: CRp density')
            else:
                if not self._silent: print('----- Plot not done: CRp density (because X_crp_E["X"] <= 0)')
                        
            # Cosmic ray proton to thermal energy
            if self._X_crp_E['X'] > 0:
                rad, prof = self.get_crp_to_thermal_energy_profile(radius, Emin=Epmin, Emax=Epmax)
                profile(radius, angle, prof.to('adu'), mydir+'/PLOT_PROF_crp_fraction.pdf',
                        label='CRp to thermal energy $X_{CRp}$', R500=self._R500)
                if not self._silent: print('----- Plot done: CRp/thermal energy')
            else:
                if not self._silent: print('----- Plot not done: CRp/thermal energy (because X_crp_E["X"] <= 0)')
                    
            # Cosmic ray electrons secondaries
            if self._X_crp_E['X'] > 0:
                rad, prof = self.get_density_cre2_profile(radius, Emin=Eemin, Emax=Eemax, Energy_density=False)
                profile(radius, angle, prof.to('cm-3'), mydir+'/PLOT_PROF_cre2_density.pdf',
                        label='CRe2 density (cm$^{-3}$)', R500=self._R500)
                if not self._silent: print('----- Plot done: CRe2 density')
            else:
                if not self._silent: print('----- Plot not done: CRe2 density (because X_crp_E["X"] <= 0)')
                    
            # Cosmic ray electrons primaries
            if self._X_cre1_E['X'] > 0:
                rad, prof = self.get_density_cre1_profile(radius, Emin=Eemin, Emax=Eemax, Energy_density=False)
                profile(radius, angle, prof.to('cm-3'), mydir+'/PLOT_PROF_cre1_density.pdf',
                        label='CRe1 density (cm$^{-3}$)', R500=self._R500)
                if not self._silent: print('----- Plot done: CRe1 density')
            else:
                if not self._silent: print('----- Plot not done: CRe1 density (because X_cre1_E["X"] <= 0)')
                    
            # Cosmic ray electron to thermal energy
            if self._X_cre1_E['X'] > 0:
                rad, prof = self.get_cre1_to_thermal_energy_profile(radius, Emin=Eemin, Emax=Eemax)
                profile(radius, angle, prof.to('adu'), mydir+'/PLOT_PROF_cre1_fraction.pdf',
                        label='CRe1 to thermal energy $X_{CRe1}$', R500=self._R500)
                if not self._silent: print('----- Plot done: CRe1/thermal energy')
            else:
                if not self._silent: print('----- Plot not done: CRe1/thermal energy (because X_cre1_E["X"] <= 0)')
                    
            # Gamma ray profile
            if self._X_crp_E['X'] > 0:
                rad, prof = self.get_gamma_profile(radius, Emin=Egmin, Emax=Egmax, Energy_density=False)
                profile(radius, angle, prof.to('cm-2 s-1 sr-1'), mydir+'/PLOT_PROF_gamma.pdf',
                        label='$\\gamma$-ray, '+Egstrlim+' (cm$^{-2}$ s$^{-1}$ sr$^{-1}$)', R500=self._R500)
                if not self._silent: print('----- Plot done: gamma surface brightness profile')
            else:
                if not self._silent: print('----- Plot not done: gamma surface brightness profile (because X_crp_E["X"] <= 0)')
                    
            # Gamma ray integrated flux profile
            if self._X_crp_E['X'] > 0:
                prof = self.get_gamma_flux(Emin=Egmin, Emax=Egmax, Rmax=radius, Energy_density=False, type_integral='spherical')
                profile(radius, angle, prof.to('cm-2 s-1'), mydir+'/PLOT_PROF_gammaF.pdf',
                        label='$\\gamma$-ray flux (<R, sph), '+Egstrlim+' (cm$^{-2}$ s$^{-1}$)', R500=self._R500)
                if not self._silent: print('----- Plot done: gamma integrated (R) flux')
            else:
                if not self._silent: print('----- Plot not done: gamma integrated (R) flux (because X_crp_E["X"] <= 0)')
                        
            # neutrino profile
            if self._X_crp_E['X'] > 0:
                rad, prof = self.get_neutrino_profile(radius, Emin=Egmin, Emax=Egmax, Energy_density=False, flavor='all')
                profile(radius, angle, prof.to('cm-2 s-1 sr-1'), mydir+'/PLOT_PROF_neutrino.pdf',
                        label='$\\nu$, '+Egstrlim+' (cm$^{-2}$ s$^{-1}$ sr$^{-1}$)', R500=self._R500)
                if not self._silent: print('----- Plot done: neutrino surface brightness profile')
            else:
                if not self._silent: print('----- Plot not done: neutrino surface brightness profile (because X_crp_E["X"] <= 0)')
                    
            # neutrino integrated flux profile
            if self._X_crp_E['X'] > 0:            
                prof = self.get_neutrino_flux(Emin=Egmin, Emax=Egmax, Rmax=radius, Energy_density=False,
                                              type_integral='spherical', flavor='all')
                profile(radius, angle, prof.to('cm-2 s-1'), mydir+'/PLOT_PROF_neutrinoF.pdf',
                        label='$\\nu$ flux (<R, sph), '+Egstrlim+' (cm$^{-2}$ s$^{-1}$)', R500=self._R500)
                if not self._silent: print('----- Plot done: neutrino integrated (R) flux')
            else:
                if not self._silent: print('----- Plot not done: neutrino integrated (R) (because X_crp_E["X"] <= 0)')
                    
            # IC profile
            if self._X_crp_E['X'] > 0 or self._X_cre1_E['X'] > 0:
                rad, prof = self.get_ic_profile(radius, Emin=Egmin, Emax=Egmax, Energy_density=False)
                profile(radius, angle, prof.to('cm-2 s-1 sr-1'), mydir+'/PLOT_PROF_InverseCompton.pdf',
                        label='IC, '+Egstrlim+' (cm$^{-2}$ s$^{-1}$ sr$^{-1}$)', R500=self._R500)
                if not self._silent: print('----- Plot done: IC surface brightness profile')
            else:
                if not self._silent: print('----- Plot not done: IC surface brightness profile (because X_crp_E["X"] <= 0 & X_cre1_E["X"] <= 0)')
                    
            # IC integrated flux profile
            if self._X_crp_E['X'] > 0 or self._X_cre1_E['X'] > 0:
                prof = self.get_ic_flux(Emin=Egmin, Emax=Egmax, Rmax=radius, Energy_density=False, type_integral='spherical')
                profile(radius, angle, prof.to('cm-2 s-1'), mydir+'/PLOT_PROF_InverseComptonF.pdf',
                        label='IC flux (<R, sph), '+Egstrlim+' (cm$^{-2}$ s$^{-1}$)', R500=self._R500)
                if not self._silent: print('----- Plot done: IC integrated (R) flux')
            else:
                if not self._silent: print('----- Plot not done: IC integrated (R) flux (because X_crp_E["X"] <= 0 & X_cre1_E["X"] <= 0)')
                    
            # Synchrotron profile
            if self._X_crp_E['X'] > 0 or self._X_cre1_E['X'] > 0:            
                rad, prof = self.get_synchrotron_profile(radius, freq0=freq0)
                profile(radius, angle, prof.to('Jy sr-1'), mydir+'/PLOT_PROF_synchrotron.pdf',
                        label='Synchrotron, '+str(freq0)+' (Jy sr$^{-1}$)', R500=self._R500)
                if not self._silent: print('----- Plot done: Synchrotron surface brightness profile')
            else:
                if not self._silent: print('----- Plot not done: Synchrotron surface brightness profile (because X_crp_E["X"] <= 0 & X_cre1_E["X"] <= 0)')
                    
            # Synchrotron integrated flux profile
            if self._X_crp_E['X'] > 0 or self._X_cre1_E['X'] > 0:
                prof = self.get_synchrotron_flux(freq0=freq0, Rmax=radius, type_integral='spherical')
                profile(radius, angle, prof.to('Jy'), mydir+'/PLOT_PROF_synchrotronF.pdf',
                        label='Synchrotron flux (<R, sph), '+str(freq0)+' (Jy)', R500=self._R500)
                if not self._silent: print('----- Plot done: Synchrotron integrated (R) flux')
            else:
                if not self._silent: print('----- Plot not done: Synchrotron integrated (R) flux (because X_crp_E["X"] <= 0 & X_cre1_E["X"] <= 0)')
                
            # Compton parameter
            rad, prof = self.get_sz_profile(radius, Compton_only=True)
            profile(radius, angle, prof.to('adu'), mydir+'/PLOT_PROF_SZ.pdf',
                    label='y Compton', R500=self._R500)
            if not self._silent: print('----- Plot done: SZ Compton')
            
            # Spherically integrated Compton
            prof = self.get_sz_flux(Rmax=radius, Compton_only=True, type_integral='spherical')
            profile(radius, angle, prof.to('kpc2'), mydir+'/PLOT_PROF_SZF.pdf',
                    label='Y spherical (kpc$^2$)', R500=self._R500)
            if not self._silent: print('----- Plot done: SZ integrated Compton (spherical)')

            if os.path.exists(self._output_dir+'/XSPEC_table.txt'):

                # Sx profile
                rad, prof = self.get_xray_profile(radius, output_type='C')
                profile(radius, angle, prof.to('s-1 cm-2 sr-1'), mydir+'/PLOT_PROF_X.pdf',
                        label='X-ray (s$^{-1}$ cm$^{-2}$ sr$^{-1}$)', R500=self._R500)
                if not self._silent: print('----- Plot done: Xray surface brightness')

                # Spherically integrated Xray flux
                prof = self.get_xray_flux(Rmax=radius, type_integral='spherical', output_type='C')
                profile(radius, angle, prof.to('s-1 cm-2'), mydir+'/PLOT_PROF_XF.pdf',
                        label='$F_X$ spherical (s$^{-1}$ cm$^{-2}$)', R500=self._R500)
                if not self._silent: print('----- Plot done: Xray integrated flux (spherical)')
                                
            else:
                print('!!! WARNING: XSPEC_table.txt not generated, skip Xray flux and Sx')
                
        #---------- Spectra
        if 'all' in prod_list or 'spectra' in prod_list:
            # CR protons
            if self._X_crp_E['X'] > 0:
                eng, spec = self.get_crp_spectrum(energy, Rmax=Rmax)
                spectra(energy, (energy/const.h).to('GHz'), spec.to('GeV-1'),
                        mydir+'/PLOT_SPEC_CRp.pdf', label='Volume integrated CRp (GeV$^{-1}$)')
                if not self._silent: print('----- Plot done: CRp spectrum')
            else:
                if not self._silent: print('----- Plot not done: CRp spectrum (because X_crp_E["X"] <= 0)')
                
            # CR electrons secondaries
            if self._X_crp_E['X'] > 0:
                eng, spec = self.get_cre2_spectrum(energy, Rmax=Rmax)
                spectra(energy, (energy/const.h).to('GHz'), spec.to('GeV-1'),
                        mydir+'/PLOT_SPEC_CRe2.pdf', label='Volume integrated CRe2 (GeV$^{-1}$)')
                if not self._silent: print('----- Plot done: CRe2 spectrum')
            else:
                if not self._silent: print('----- Plot not done: CRe2 spectrum (because X_crp_E["X"] <= 0)')
                
            # CR electrons primaries
            if self._X_cre1_E['X'] > 0:
                eng, spec = self.get_cre1_spectrum(energy, Rmax=Rmax)
                spectra(energy, (energy/const.h).to('GHz'), spec.to('GeV-1'),
                        mydir+'/PLOT_SPEC_CRe1.pdf', label='Volume integrated CRe1 (GeV$^{-1}$)')
                if not self._silent: print('----- Plot done: CRe1 spectrum')
            else:
                if not self._silent: print('----- Plot not done: CRe1 spectrum (because X_cre1_E["X"] <= 0)')
                
            # gamma
            if self._X_crp_E['X'] > 0:
                eng, spec = self.get_gamma_spectrum(energy, Rmax=Rmax, type_integral='spherical')
                spectra(energy, (energy/const.h).to('GHz'), (energy**2*spec).to('GeV cm-2 s-1'),
                        mydir+'/PLOT_SPEC_gamma.pdf',
                        label='$F_{\\gamma}$(<R, sph) (GeV cm$^{-2}$ s$^{-1}$)')
                if not self._silent: print('----- Plot done: gamma spectrum')
            else:
                if not self._silent: print('----- Plot not done: gamma spectrum (because X_crp_E["X"] <= 0)')
                
            # Gamma integrated flux spectrum
            if self._X_crp_E['X'] > 0:
                spec = self.get_gamma_flux(Emin=energy, Emax=Egmax, Rmax=Rmax, Energy_density=False, type_integral='spherical')
                spectra(energy, (energy/const.h).to('GHz'), spec.to('cm-2 s-1'),
                        mydir+'/PLOT_SPEC_gammaF.pdf', label='$\\gamma$-ray flux (>E, sph) (cm$^{-2}$ s$^{-1}$)')
                if not self._silent: print('----- Plot done: gamma integrated flux (E)')
            else:
                if not self._silent: print('----- Plot not done: gamma integrated flux (E) (because X_crp_E["X"] <= 0)')
                
            # neutrino
            if self._X_crp_E['X'] > 0:
                eng, spec = self.get_neutrino_spectrum(energy, Rmax=Rmax, type_integral='spherical', flavor='all')
                spectra(energy, (energy/const.h).to('GHz'), (energy**2*spec).to('GeV cm-2 s-1'),
                        mydir+'/PLOT_SPEC_neutrino.pdf', label='$F_{\\nu}$(<R, sph) (GeV cm$^{-2}$ s$^{-1}$)')
                if not self._silent: print('----- Plot done: neutrino spectrum')
            else:
                if not self._silent: print('----- Plot not done: neutrino spectrum (because X_crp_E["X"] <= 0)')
                
            # neutrino integrated flux spectrum
            if self._X_crp_E['X'] > 0:
                spec = self.get_neutrino_flux(Emin=energy, Emax=Egmax, Rmax=Rmax, Energy_density=False, type_integral='spherical',
                                              flavor='all')
                spectra(energy, (energy/const.h).to('GHz'), spec.to('cm-2 s-1'),
                        mydir+'/PLOT_SPEC_neutrinoF.pdf',label='$\\nu$ flux (>E, sph) (cm$^{-2}$ s$^{-1}$)')
                if not self._silent: print('----- Plot done: neutrino integrated flux (E)')
            else:
                if not self._silent: print('----- Plot not done: neutrino integrated flux (E) (because X_crp_E["X"] <= 0)')
                
            # IC
            if self._X_crp_E['X'] > 0 or self._X_cre1_E['X'] > 0:
                eng, spec = self.get_ic_spectrum(energy, Rmax=Rmax, type_integral='spherical')
                spectra(energy, (energy/const.h).to('GHz'), (energy**2*spec).to('GeV cm-2 s-1'),
                        mydir+'/PLOT_SPEC_InverseCompton.pdf',label='$F_{IC}$(<R, sph) (GeV cm$^{-2}$ s$^{-1}$)')
                if not self._silent: print('----- Plot done: IC spectrum')
            else:
                if not self._silent: print('----- Plot not done: IC spectrum (because X_crp_E["X"] <= 0 & X_cre1_E["X"] <= 0)')
                
            # IC integrated flux spectrum
            if self._X_crp_E['X'] > 0 or self._X_cre1_E['X'] > 0:
                spec = self.get_ic_flux(Emin=energy, Emax=Egmax, Rmax=Rmax, Energy_density=False, type_integral='spherical')
                spectra(energy, (energy/const.h).to('GHz'), spec.to('cm-2 s-1'), mydir+'/PLOT_SPEC_InverseComptonF.pdf',
                        label='IC flux (>E, sph) (cm$^{-2}$ s$^{-1}$)')
                if not self._silent: print('----- Plot done: IC integrated flux (E)')
            else:
                if not self._silent: print('----- Plot not done: IC integrated flux (E) (because X_crp_E["X"] <= 0 & X_cre1_E["X"] <= 0)')
                
            # Synchrotron
            if self._X_crp_E['X'] > 0 or self._X_cre1_E['X'] > 0:
                freq, spec = self.get_synchrotron_spectrum(frequency, Rmax=Rmax, type_integral='spherical')
                spectra((freq*const.h).to('eV'), freq, spec.to('Jy'), mydir+'/PLOT_SPEC_Synchrotron.pdf',
                        label='$F_{synch}$(<R, sph) (Jy)')
                if not self._silent: print('----- Plot done: Synchrotron spectrum')
            else:
                if not self._silent: print('----- Plot not done: Synchrotron spectrum (because X_crp_E["X"] <= 0 & X_cre1_E["X"] <= 0)')
                
            # SZ
            freq, spec = self.get_sz_spectrum(frequency, Rmax=Rmax, type_integral='spherical', Compton_only=False)
            spectra((freq*const.h).to('eV'), freq, np.abs(spec.to('Jy')), mydir+'/PLOT_SPEC_SZ.pdf',
                    label='$|F_{SZ}|$(<R, sph) (Jy)')
            if not self._silent: print('----- Plot done: SZ spectrum')

            # Xray
            if os.path.exists(mydir+'/XSPEC_table.txt'):
                engX, spec = self.get_xray_spectrum(energyX, Rmax=Rmax, type_integral='spherical', output_type='C')
                spectra(engX.to('keV'), (engX/const.h).to('GHz'), spec.to('cm-2 s-1 keV-1'), mydir+'/PLOT_SPEC_X.pdf',
                        label='$S_{X}$(<R, sph) (cm$^{-2}$ s$^{-1}$ keV${-1}$)')
                if not self._silent: print('----- Plot done: Xray spectrum')
            else:
                print('!!! WARNING: XSPEC_table.txt not generated, skip Xray spectrum')
                
        #---------- Map
        if 'all' in prod_list or 'map' in prod_list:
            header = self.get_map_header()

            # gamma
            if self._X_crp_E['X'] > 0:
                image = self.get_gamma_map(Emin=Egmin, Emax=Egmax, Energy_density=False,
                                           Normalize=False).to_value('cm-2 s-1 sr-1')
                maps(image, header, mydir+'/PLOT_MAP_gamma.pdf',
                     label='$\\gamma$-ray, '+Egstrlim+' (cm$^{-2}$ s$^{-1}$ sr$^{-1}$)',
                     coord=self._coord, theta_500=self._theta500, theta_trunc=self._theta_truncation, logscale=True)
                if not self._silent: print('----- Plot done: gamma map')
            else:
                if not self._silent: print('----- Plot not done: gamma map (because X_crp_E["X"] <= 0)')
            
            # neutrino
            if self._X_crp_E['X'] > 0:
                image = self.get_neutrino_map(Emin=Egmin, Emax=Egmax, Energy_density=False,
                                              Normalize=False, flavor='all').to_value('cm-2 s-1 sr-1')
                maps(image, header, mydir+'/PLOT_MAP_neutrino.pdf', label='$\\nu$, '+Egstrlim+' (cm$^{-2}$ s$^{-1}$ sr$^{-1}$)',
                     coord=self._coord, theta_500=self._theta500, theta_trunc=self._theta_truncation, logscale=True)
                if not self._silent: print('----- Plot done: neutrino map')
            else:
                if not self._silent: print('----- Plot not done: neutrino map (because X_crp_E["X"] <= 0)')
                
            # IC
            if self._X_crp_E['X'] > 0 or self._X_cre1_E['X'] > 0:
                image = self.get_ic_map(Emin=Egmin, Emax=Egmax, Energy_density=False, Normalize=False).to_value('cm-2 s-1 sr-1')
                maps(image, header, mydir+'/PLOT_MAP_inverseCompton.pdf', label='IC, '+Egstrlim+' (cm$^{-2}$ s$^{-1}$ sr$^{-1}$)',
                     coord=self._coord, theta_500=self._theta500, theta_trunc=self._theta_truncation, logscale=True)
                if not self._silent: print('----- Plot done: IC map')
            else:
                if not self._silent: print('----- Plot not done: IC map (because X_crp_E["X"] <= 0 & X_cre1_E["X"] <= 0)')
                
            # Synchrotron
            if self._X_crp_E['X'] > 0 or self._X_cre1_E['X'] > 0:
                image = self.get_synchrotron_map(freq0=freq0, Normalize=False).to_value('Jy sr-1')
                maps(image, header, mydir+'/PLOT_MAP_synchrotron.pdf', label='Synchrotron, '+str(freq0)+' (Jy sr$^{-1}$)',
                     coord=self._coord, theta_500=self._theta500, theta_trunc=self._theta_truncation, logscale=True)
                if not self._silent: print('----- Plot done: Synchrotron map')
            else:
                if not self._silent: print('----- Plot not done: Synchrotron map (because X_crp_E["X"] <= 0 & X_cre1_E["X"] <= 0)')
                
            # SZ ymap
            image = self.get_sz_map(Compton_only=True, Normalize=False).to_value('adu')
            maps(image*1e6, header, mydir+'/PLOT_MAP_SZy.pdf',
                 label='Compton parameter $\\times 10^{6}$', coord=self._coord, theta_500=self._theta500,
                 theta_trunc=self._theta_truncation, logscale=True)
            if not self._silent: print('----- Plot done: SZ ymap')

            # Xray
            if os.path.exists(self._output_dir+'/XSPEC_table.txt'):
                
                image = self.get_xray_map(output_type='C').to_value('s-1 cm-2 sr-1')
                maps(image, header, mydir+'/PLOT_MAP_X.pdf', label='X-ray (s$^{-1}$ cm$^{-2}$ sr$^{-1}$)',
                     coord=self._coord, theta_500=self._theta500, theta_trunc=self._theta_truncation, logscale=True)
                if not self._silent: print('----- Plot done: Xray map')
                
            else:
                print('!!! WARNING: XSPEC_table.txt not generated, skip PLOT_MAP_X')
                

