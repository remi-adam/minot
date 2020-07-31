""" 
This script gather several functions which are related to the electron
energy loss via different processes.
"""

import numpy as np
import astropy.units as u
from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo

#===================================================
#========== Synchrotron loss
#===================================================
def dEdt_sync(energy, B):
    """
    Compute loss via synchrotron radiation.
    From M. Longair, High Energy Astrophysics

    Parameters
    ----------
    - energy (quantity): energy array homogeneous to GeV
    - B (quantity): magnetic field strength homogeneous to Gauss

    Outputs
    --------
    - Energy loss (GeV/s)
    """
    
    gamma = (energy/(const.m_e*const.c**2)).to('')
    w_neg = gamma <= 1
    gamma[w_neg] = 2 # set negative values to 2, and will set result to 0 there
    beta = np.sqrt(1.0-1.0/gamma**2)
    
    dEdt = 4.0/3.0 * const.sigma_T * const.c * beta**2 * gamma**2 * B**2/(2*const.mu0)

    # Energy cannot be lower than rest mass
    dEdt[w_neg] = 0
    
    return dEdt.to('GeV s-1')

#===================================================
#========== Inverse Compton loss
#===================================================
def dEdt_ic(energy, redshift):
    """
    Compute loss via inverse Compton.
    From M. Longair, High Energy Astrophysics

    Parameters
    ----------
    - energy (quantity): energy array homogeneous to GeV
    - redshift (float): cluster redshift

    Outputs
    --------
    - Energy loss (GeV/s)
    """
    
    gamma = (energy/(const.m_e*const.c**2)).to('')
    w_neg = gamma <= 1
    gamma[w_neg] = 2 # set negative values to 2, and will set result to 0 there
    beta = np.sqrt(1.0-1.0/gamma**2)

    Ucmb = 8*np.pi**5 * (const.k_B*cosmo.Tcmb0*(1+redshift))**4 / 15.0 / (const.h*const.c)**3
    
    dEdt = 4.0/3.0*const.sigma_T*const.c * beta**2*gamma**2 * Ucmb

    # Energy cannot be lower than rest mass
    dEdt[w_neg] = 0
    
    return dEdt.to('GeV s-1')

#===================================================
#========== Coulomb loss (ionisation)
#===================================================
def dEdt_coul(energy, n_e):
    """
    Compute loss via Coulomb/ionisation.
    From Pinzke et al. (2017), which is based on Gould 1972.
    Check adapting Longair Bethe Block formula

    Parameters
    ----------
    - energy (quantity): energy array homogeneous to GeV
    - n_e (quantity): the number density of ambient electrons

    Outputs
    --------
    - Energy loss (GeV/s)
    """

    wbad = (n_e <= 0)
    
    gamma = (energy/(const.m_e*const.c**2)).to('')
    w_neg = gamma <= 1
    gamma[w_neg] = 2 # set negative values to 2, and will set result to 0 there

    beta = np.sqrt(1.0-1.0/gamma**2)
    omega_p = np.sqrt((const.e.value*const.e.unit)**2 * n_e / const.m_e / const.eps0) # Plasma frequency
    omega_p[wbad] = np.nan
    
    f1 = np.log((const.m_e*const.c**2*beta*np.sqrt(gamma-1))/(const.hbar*omega_p))
    f2 = np.log(2)*(beta**2/2 + 1/gamma)
    f3 = ((gamma-1.0)/4.0/gamma)**2
    
    dEdt = 3.0/4.0*const.sigma_T * n_e * const.m_e*const.c**3/beta * 2*(f1-f2+f3+1.0/2)

    # Energy cannot be lower than rest mass
    dEdt[w_neg] = 0
    
    dEdt[wbad] = 0 
    
    return dEdt.to('GeV s-1')


#===================================================
#========== Bremsstrahlung loss
#===================================================
def dEdt_brem(energy, n_e, ne2nth=1.071):
    """
    Compute loss Bremsstrahlung from Blumenthal & Gould 1970

    Parameters
    ----------
    - energy (quantity): energy array homogeneous to GeV
    - n_e (quantity): the number density of ambient electrons
    - ne2nth (float): corresponds to conversion from n_e to n_th, being
    equal to mu_e/mu_p + mu_e/mu_He

    Outputs
    --------
    - Energy loss (GeV/s)
    """

    alpha = ((const.e.value*const.e.unit)**2 / (4*np.pi*const.eps0*const.hbar*const.c)).to_value('')
    r0 = (const.e.value*const.e.unit)**2 / (const.m_e*const.c**2) / (4*np.pi*const.eps0)
    gamma = (energy/(const.m_e*const.c**2)).to('')
    w_neg = gamma <= 1
    gamma[w_neg] = 2 # set negative values to 2, and will set result to 0 there
    dEdt = 8*alpha*r0**2*energy*const.c* ne2nth*n_e * (np.log(2*gamma) - 1.0/3)
    
    return dEdt.to('GeV s-1')


#===================================================
#========== Total losses
#===================================================
def dEdt_tot(energy, radius=None, n_e=1*u.cm**-3, B=1*u.uG, redshift=0.0, ne2nth=1.071):
    """
    Compute loss from brem + Coul + IC + Sync

    Parameters
    ----------
    - energy (quantity): energy array homogeneous to GeV
    - radius (quantity) : array of radius to be provided if physial
    parameters are position dependant
    - n_e (quantity): the number density of ambient electrons
    - B (quantity): magnetic field strength homogeneous to Gauss
    - redshift (float): cluster redshift
    - ne2nth (float): corresponds to conversion from n_e to n_th, being
    equal to mu_e/mu_p + mu_e/mu_He

    Outputs
    --------
    - Energy loss (GeV/s): 2d array with one axis corresponding to radius and other one to energy
    """

    #========== In case energy, n_e, B are just 1D array
    if radius is None:        
        e_grid = energy
        n_grid = n_e
        B_grid = B
    else:
        if type(n_e.value) != np.ndarray or type(B.value) != np.ndarray or type(radius.value) != np.ndarray:
            raise ValueError("The type of radius, n_e and B should be ndarray if radius is provided")
        if len(n_e) != len(radius) or len(B) != len(radius):
            raise ValueError("The shape of n_e and B are not consistent with radius")

        N_rad = len(radius)
        N_eng = len(energy)
        
        e_grid = (np.tile(energy, [N_rad,1])).T
        n_grid = (np.tile(n_e, [N_eng, 1]))
        B_grid = (np.tile(B, [N_eng, 1]))

    #========== Compute
    dEdt1 = dEdt_sync(e_grid, B_grid)
    dEdt2 = dEdt_ic(e_grid, redshift)
    dEdt3 = dEdt_brem(e_grid, n_grid, ne2nth=ne2nth)
    dEdt4 = dEdt_coul(e_grid, n_grid)

    dEdt = dEdt1 + dEdt2 + dEdt3 + dEdt4    

    return dEdt.to('GeV s-1')


