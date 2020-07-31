""" 
This script gather several functions which are related to the spectral 
properties of galaxy clusters. It is mainly written following on the 
paper by Kelner et al. (2006).
"""

import numpy as np
import copy
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from astropy import units as u 
from astropy import constants as const
import matplotlib.pyplot as plt
from ebltable.tau_from_model import OptDepth

from minot.ClusterTools import cluster_profile


#===================================================
#========== Compute the pp interaction thershold
#===================================================
def pp_pion_kinematic_energy_threshold():
    """
    Compute the kinematic energy threshold of pp collision -> pions

    Parameters
    ----------

    Outputs
    --------
    - Energy threshold (GeV)
    """

    m_pi0 = 0.1349766  # GeV, from PDG
    m_p = (const.m_p * const.c**2).to_value('GeV')
    E_th = m_p + 2*m_pi0 + m_pi0**2 / 2.0 / m_p
    
    return E_th


#===================================================
#========== PowerLaw model
#===================================================
def powerlaw_model(energy_gev, k0, index, E0=1.0):
    """
    Compute a PowerLaw spectrum

    Parameters
    ----------
    - energy_GeV: scalar or vector
    - k0 : normalization
    - E0 : pivot energy (GeV)
    - index : spectral index

    Outputs
    --------
    - spectrum
    """

    return k0 * (energy_gev/E0)**(-index)


#===================================================
#========== Exponential Cutoff PowerLaw model
#===================================================
def exponentialcutoffpowerlaw_model(energy_gev, k0, index, Ecut, E0=1.0):
    """
    Compute a PowerLaw spectrum with exponential

    Parameters
    ----------
    - energy_GeV: scalar or vector
    - k0 : normalization
    - E0 : pivot energy (GeV)
    - index : spectral index
    - Ecut : cutoff energy (GeV)

    Outputs
    --------
    - spectrum
    """
    
    return k0 * (energy_gev/E0)**(-index) * np.exp(-energy_gev/Ecut)


#===================================================
#========== Momentum Space PowerLaw model
#===================================================

def momentumpowerlaw_model(energy_gev, k0, index,
                           E0=1.0,
                           mass=const.m_e*const.c**2):
    """
    Compute a PowerLaw spectrum in momentum space

    Parameters
    ----------
    - energy_GeV: scalar or vector
    - k0 : normalization
    - index : spectral index
    - E0 : pivot energy (GeV)
    - mass (qauntity): the  mass of the considered particle

    Outputs
    --------
    - spectrum
    """

    E_gev = copy.copy(energy_gev)
    wbad = energy_gev <= mass.to_value('GeV')
    E_gev[wbad] = 2*mass.to_value('GeV') # avoid error, but should be set to zero

    if E0 <= mass.to_value('GeV'):
        raise ValueError('The pivot energy E0 should be larger than the particle mass')
    
    P0 = np.sqrt(E0**2 - (mass.to_value('GeV'))**2) / const.c.to_value('m/s')
    
    momentum = np.sqrt(E_gev**2 - (mass.to_value('GeV'))**2) / const.c.to_value('m/s')

    fP = k0 * (momentum/P0)**(-index)
    dP = (E_gev/const.c.to_value('m/s'))  / np.sqrt(E_gev**2 - (mass.to_value('GeV'))**2)
     
    SE = fP*dP
    
    SE[wbad] = 0
    
    return SE


#===================================================
#========== Initial injection (Jaffe Perola) model
#===================================================

def initial_injection_model(energy_gev, k0, index, Ebreak,
                            E0=1.0):
    """
    Compute a JP spectrum. See Jaffe and Perola (1973),
    and Turner 2017 (+Pacholczyk 1970; Longair 2010)

    Parameters
    ----------
    - energy_GeV: scalar or vector
    - k0 : normalization
    - index : spectral index
    - E0 : pivot energy (GeV)
    - Ebreak : break energy (GeV)

    Outputs
    --------
    - spectrum
    """

    E_gev = copy.copy(energy_gev)
    wbad = energy_gev > Ebreak
    E_gev[wbad] = Ebreak/2.0 # avoid error, but should be set to zero

    S = k0 * (E_gev/E0)**(-index) * (1-E_gev/Ebreak)**(index-2)
    S[wbad] = 0
    
    return S


#===================================================
#========== Continuous Injection model
#===================================================

def continuous_injection_model(energy_gev, k0, index, Ebreak,
                               E0=1.0):
    """
    Compute a CI spectrum. See Pacholczyk 1970,
    and Turner 2017

    Parameters
    ----------
    - energy_GeV: scalar or vector
    - k0 : normalization
    - index : spectral index
    - E0 : pivot energy (GeV)
    - Ebreak : break energy (GeV)

    Outputs
    --------
    - spectrum
    """

    energy_gev_1 = copy.copy(energy_gev)
    energy_gev_2 = copy.copy(energy_gev)
    w1 = energy_gev >= Ebreak
    w2 = energy_gev < Ebreak
    energy_gev_2[w1] = Ebreak/2.0

    f1 = k0 * (energy_gev_1/E0)**(-index-1)
    f2 = 1 - (1-energy_gev_2/Ebreak)**(index-1)
    S = energy_gev*0
    S[w1] = f1[w1]
    S[w2] = (f1*f2)[w2]
    
    return S


#===================================================
#========== Integral power law
#===================================================
def get_integral_powerlaw_model(Emin, Emax, k0, index, E0=1.0):
    """
    Compute the enery integral :
    \int_Emin^Emax f(E) dE
    for f(E) a power law

    Parameters
    ----------
    - Emin (GeV): the lower bound
    - Emax (GeV): the upper bound
    - k0 : normalization
    - E0 : pivot energy (GeV)
    - index : spectral index

    Outputs
    --------
    - The integrated function
    
    """

    if Emin > Emax:
        raise TypeError("Emin is larger than Emax")

    output = k0 * E0**(1-index) / (1-index) * ( (Emax/E0)**(1-index) - (Emin/E0)**(1-index))
    
    return output


#===================================================
#========== Integrate a model
#===================================================
def get_integral_any_model(energy, f, Emin, Emax, Npt=1000):
    """
    Compute the enery integral :
    \int_Emin^Emax f(E) dE

    Parameters
    ----------
    - energy (GeV): energy vector associated to f
    - f  : the function to integrate
    - Emin (GeV): the lower bound
    - Emax (GeV): the upper bound

    Outputs
    --------
    - The integrated function
    
    """

    if Emin > Emax:
        raise TypeError("Emin is larger than Emax")

    if Emin < 0.999*np.amin(energy):
        print('!!!!!!!!!! WARNING: you try to integrate below function limits !!!!!!!!!!')
    if Emax > 1.0001*np.amax(energy):
        print('!!!!!!!!!! WARNING: you try to integrate beyond function limits !!!!!!!!!!')
    
    # interpolate to the location of interest
    eng_i = np.logspace(np.log10(Emin), np.log10(Emax), num=Npt)

    itpl = interpolate.interp1d(energy, f, kind='cubic', fill_value='extrapolate')
    f_i = itpl(eng_i)

    # integrate
    output = np.trapz(f_i, eng_i)

    return output


#===================================================
#========== Heaviside function
#===================================================
def heaviside(x):
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


#===================================================
#========== Proton-proton cross section
#===================================================
def sigma_pp(E_p, model='Kafexhiu2014'):
    """
    Give the proton proton interaction cross-section as 
    a function of energy.

    Parameters
    ----------
    - E_p : the energy of protons in GeV

    Outputs
    --------
    - sig_pp : the cross section in mb
    """
    
    E_p = np.array(E_p)

    # constant values
    m_p = (const.m_p * const.c**2).to_value('GeV') # GeV
    E_th = pp_pion_kinematic_energy_threshold()

    # Kelner default model
    if model == 'Kelner2006':
        L = np.log(E_p * 1e-3)  # E_p should be in GeV
        sig_pp = (34.3 + 1.88 * L + 0.25 * L ** 2) * (1 - (E_th / E_p) ** 4) ** 2

    # Kafexhiu other model
    if model == 'Kafexhiu2014':
        T_th = E_th - m_p
        T_p = E_p - m_p
        L = np.log(T_p/T_th)
        sig_pp = (30.7 - 0.96*L + 0.18*L**2) * (1 - (T_th/T_p)**1.9)**3
        
    sig_pp = sig_pp * heaviside(E_p - E_th)
    
    return sig_pp


#===================================================
#========== Pariticle distribution term
#===================================================
def J_p(Ep, norm=1.0, alpha=2.0):
    """
    Defines the particle distribution of the protons as a power law.

    Parameters
    ----------
    - Ep = E_proton (GeV)
    - alpha = slope
    - norm = the normalization (at 1 GeV) in units of cm^-3 GeV-1

    Outputs
    --------
    - The particle distribution in units of 1/GeV/cm^-3
    """

    return norm*(Ep/1.0)**(-alpha)


#===================================================
#========== Pariticle distribution term multiplyied by energy
#===================================================
def J_p2(Ep, norm=1.0, alpha=2.0):
    """
    Defines the particle distribution of the protons as a power law
    
    Parameters
    ----------
    - Ep = E_proton (GeV)
    - alpha = slope
    - norm = the normalization (at 1 GeV) in units of cm^-3 GeV-1

    Outputs
    --------
    - The particle distribution times energy in units of cm^-3
    """
    return Ep*norm*(Ep/1.0)**(-alpha)


#===================================================
#========== total CR energy density
#===================================================
def get_cr_energy(norm=1.0, alpha=2.0, Emax=np.inf):
    """ 
    Compute the total energy of cosmic ray proton given a 
    power law spectrum

    Parameters
    ----------
    - norm: the normalization (GeV^-1 cm^-3)
    - alpha: the slope
    - Emax (GeV) : the maximum integration. This is important for 
    slopes <~ 2. From Pinzke et al 2010, we expect a cutoff at E~10^10 GeV.

    Outputs
    --------    
    - U_CR = the energy in GeV/cm^-3
    """

    E_th = pp_pion_kinematic_energy_threshold()    
    U_CR = integrate.quad(J_p2, E_th, Emax, args=(norm, alpha))[0] # GeV/cm^-3

    return U_CR


#===================================================
#========== total CR energy density
#===================================================
def get_ebl_absorb(energy_GeV, redshift, EBL_model):
    """ 
    Compute the EBL absorbtion.

    Parameters
    ----------
    - energy_GeV (array): photon energy in GeV
    - redshift (float): object redshift
    - EBL_model (str): the model to be used
    
    Outputs
    --------    
    - absorb = the absorbtion, to be multiplied to the spectrum
    """
    
    tau    = OptDepth.readmodel(model=EBL_model)
    absorb = np.exp(-1. * tau.opt_depth(redshift, energy_GeV*1e-3))

    # search for duplicate, because of bug in interpolation at high absorb
    minabs = np.nanmin(absorb)

    if minabs <= 1e-4:
        wmin = (absorb == minabs)
        absorb[wmin] = 0.0
    
    return absorb


#===================================================
#========== Main function
#===================================================
if __name__ == "__main__":

    print('========== Cluster spectral module ==========')
    print('   This computes the expected photon spectra')
    print('based on pp interaction and pi0 decay.')
    print('   This example compute the expected spectrum')
    print('for a Coma like cluster given some profile,')
    print('an expected CR energy content, and a spectral')
    print('parametrization of the proton energy spectrum.')
    print('=============================================')
    
    # Parameters
    kpc2cm = const.kpc.to_value('cm')     # cm/kpc
    GeV2erg = const.e.value*1e+9*1e+7     # erg/GeV
    
    D_L = 100 * 1e3        # Luminosity distance (kpc)
    U_CR_tot = 1e64        # Total CR energy (GeV)
    R200 = 1000.0          # Characteristic radius (kpc)
    alpha = 2.8            # CR spectrum slope
    n0_gas = 3e-3          # density normalization (cm^-3)
    rc = 290.0             # characteristic radius (kpc)
    beta = 2.0/3           # density slope
    
    # Compute the spectrum part for ngas=1cm^-3, nCR=1cm^-3
    Egamma = np.logspace(-2,3,100)                                   # GeV
    dN_dEdVdt1 = get_luminosity_density(Egamma, 1.0, 1.0, alpha)      # GeV^-1 s^-1 cm^-2
    dN_dEdVdt2 = get_luminosity_density_lowE(Egamma, 1.0, 1.0, alpha) # GeV^-1 s^-1 cm^-2

    # Compute the radial component    
    volume_term1 = cluster_profile.get_volume_beta_model(R200, 1.0, rc, beta)   # volume integral of beta model (kpc^3)
    n0_p = U_CR_tot / (get_cr_energy(1.0, alpha) * volume_term1*kpc2cm**3)   # normalize CR density (cm^-3)
    volume_term2 = cluster_profile.get_volume_beta_model(R200, 1.0, rc, beta*2) # volume integral of beta model squared (kpc^3)
    distance_term = 4 * np.pi * (D_L*kpc2cm)**2                                 # flux to luminosity (cm^2)

    dN_dEdSdt1 = dN_dEdVdt1 * n0_gas * n0_p * volume_term2*kpc2cm**3 / distance_term #GeV^-1 s^-1 cm^-2 
    dN_dEdSdt2 = dN_dEdVdt2 * n0_gas * n0_p * volume_term2*kpc2cm**3 / distance_term
    
    # Plot the expected spectra
    fig1 = plt.figure(1)
    plt.loglog(Egamma, Egamma**2*dN_dEdSdt1 * GeV2erg, label='Valid for high energy')
    plt.loglog(Egamma, Egamma**2*dN_dEdSdt2 * GeV2erg, label='Low energy $\delta$ limit')
    plt.xlabel('Photon energy (GeV)')
    plt.ylabel('$E^2 dN/dE$ (erg/s/cm$^2$)')
    plt.legend(loc='upper right')
    plt.show()
