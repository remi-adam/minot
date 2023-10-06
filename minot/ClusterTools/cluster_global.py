""" 
This script gather several functions which are related to the global properties
of galaxy clusters.
"""

import numpy as np
import astropy.cosmology
import astropy.constants as cst
import astropy.units as u
from scipy.optimize import brentq
import matplotlib.pyplot as plt

#===================================================
#========== get M_delta from R_delta
#===================================================
def Mdelta_to_Rdelta(M_delta, redshift, delta=500, cosmo=astropy.cosmology.Planck15):
    """
    Compute a R_delta, delta being the overdensity wrt the local background

    Parameters
    ----------
    - M_delta (Msun): array

    Outputs
    --------
    - R_delta (kpc): array

    """
    
    # local density
    rho = delta*3.0*cosmo.H(redshift)**2 / 8.0 / np.pi / cst.G

    # Radius
    R_delta = ((3.0*M_delta*u.Msun/4.0/np.pi/rho)**(1.0/3)).to_value('kpc')

    return R_delta

#===================================================
#========== get r500 from m500
#===================================================
def Rdelta_to_Mdelta(R_delta, redshift, delta=500, cosmo=astropy.cosmology.Planck15):
    """
    Compute a M500

    Parameters
    ----------
    - M_delta (Msun): array

    Outputs
    --------
    - R_delta (kpc): array

    """
    
    # local density
    rho = delta*3.0*cosmo.H(redshift)**2 / 8.0 / np.pi / cst.G

    # mass
    M_delta = ((R_delta*u.kpc)**3 * rho*np.pi*4.0/3.0).to_value('Msun')

    return M_delta

#===================================================
#========== Convert M_deta1 to M_delta2 assuming NFW
#===================================================
def Mdelta1_to_Mdelta2_NFW(M_delta1, delta1=500, delta2=200, c1=3.0, redshift=0.0, cosmo=astropy.cosmology.Planck15):
    """
    Convert from, e.g., M500 to M200 assuming NFW profile. 

    Parameters
    ----------
    - Mdelta1 (Msun): array

    Outputs
    --------
    - Mdelta2 (Msun): array

    Notes
    --------
    The results do not depend on redshift or cosmology. 
    They are used internally but should cancel.

    """
    
    c1 = float(c1)
    
    R_delta1 = Mdelta_to_Rdelta(M_delta1, redshift, delta=delta1, cosmo=cosmo)   # kpc
    Rs       = R_delta1 / c1                                                     # kpc
    rho0     = M_delta1 / (4*np.pi * Rs**3) / (np.log(1 + c1) - c1/(1+c1))       # Msun/kpc^3

    # Difference between M(<R) from NFW and from M(<R) assuming R==R_delta
    def enclosed_mass_difference(radius):
        mass_nfw    = (4*np.pi * Rs**3) * rho0 * (np.log(1 + radius/Rs) - radius/(radius+Rs))
        mass_delta2 =  Rdelta_to_Mdelta(radius, redshift, delta=delta2, cosmo=cosmo)
        return  mass_nfw - mass_delta2

    # Search R_delta2 as the root of the function
    R_delta2 = brentq(enclosed_mass_difference, R_delta1*1e-5, R_delta1*1e5)
    M_delta2 = Rdelta_to_Mdelta(R_delta2, redshift, delta=delta2, cosmo=cosmo)
    
    return M_delta2


#===================================================
#========== Compute overdensity from NFW concentration
#===================================================

def concentration_to_deltac_NFW(concentration, Delta=200):
    """
    Compute the overdensity, delta_c, from the concentration 
    assuming NFW profile:
    rho_0 = delta_c(c) x rho_crit

    Parameters
    ----------
    - concentration: scalar
    - Delta: the overdensity for which the concentration is given

    Outputs
    --------
    - delta_c: scalar

    """
    
    f1 = Delta/3 * concentration**3
    f2 = 1 / (np.log(1+concentration) - concentration/(1+concentration))
    return f1 * f2


#===================================================
#========== Compute NFW concentration from overdensity
#===================================================

def deltac_to_concentration_NFW(delta_c, Delta=200):
    """
    Compute the concentration from the overdensity, delta_c,
    assuming NFW profile:
    rho_0 = delta_c(c) x rho_crit

    Parameters
    ----------
    - delta_c: scalar
    - Delta: the overdensity for which the concentration is computed

    Outputs
    --------
    - concentration: scalar

    """
    # internal function that computes the difference between input
    # overdensity and computed on from the concentration
    def deltac_diff_func(concentration, Delta, my_delta_c):
        deltac = concentration_to_deltac_NFW(concentration, Delta=Delta)
        return deltac - my_delta_c
    
    concentration = brentq(deltac_diff_func, 1e-5, 1e3, args=(Delta, delta_c))
    
    return concentration


#===================================================
#========== gNFW pressure normalization
#===================================================
def gNFW_normalization(redshift, M500, cosmo=astropy.cosmology.Planck15,
                       mu=0.59, mue=1.14, fb=0.175):
    """
    Compute a gNFW model electronic pressure normalization based on mass and redshift.
    See Arnaud et al. A&A 517, A92 (2010). This does not account to the P0 
    term, which should be multiplied to get the overall normalization.

    Parameters
    ----------
    - redshift: redshift of the cluster
    - M500 (Msun): the mass within R500 of the cluster 
    - cosmo (astropy.cosmology): cosmology module
    - mu, mue, fb (float): mean molecular weights and baryon fraction. Default values are from Arnaud+2010
    
    Outputs
    --------
    - Pnorm (keV/cm^-3): the electron pressure normalization

    """

    E_z = cosmo.efunc(redshift)
    h70 = cosmo.H0.value/70.0

    F_M = (M500/3e14*h70)**0.12
    #P500 = 1.65e-3 * E_z**(8.0/3.0) * (M500/3e14*h70)**(2.0/3.0) * h70**2
    #Pnorm = P500 * F_M
    P500 = (3/(8*np.pi)) * (500*cst.G**-(1/4)*cosmo.H(redshift)**2/2)**(4/3) * (mu/mue) * fb *(M500*u.Msun)**(2/3)
    Pnorm = P500.to_value('keV cm-3') * F_M
    return Pnorm

#===================================================
#========== Compute mean moldecular weights
#===================================================
def mean_molecular_weight(Y=0.2735, Z=0.0153):
    """
    Compute the mean molecular weight assuming recomended 
    protosolar values from Lodders 2009 (arxiv 0901.1149): Table9. 
    We use the approximation that (1+Z)/A = 1/2 for metals.

    Parameters
    ----------
    - Y : the mass fraction of Helium
    - Z : the metal mass fraction

    Outputs
    --------
    - mu_gas (float): gas mean molecular weight
    - mu_e (float): electron mean molecular weight
    - mu_p (float): proton mean molecular weight
    - mu_alpha (float): alpha mean molecular weight

    """
    
    if Y+Z > 1:
        raise ValueError('The sum Y+Z cannot exceed 1')
    
    X = np.array([1.0-Y-Z, Y, Z]) # Mass fraction
    
    mu_gas   = 1.0 / (2.0*X[0] + 3.0*X[1]/4.0 + X[2]/2.0)
    mu_e     = 1.0 / (X[0] + X[1]/2.0 + X[2]/2.0)
    
    if X[0] == 0:
        mu_p = np.nan
    else:
        mu_p = 1.0 / X[0]
        
    if X[1] == 0:
        mu_alpha = np.nan
    else:
        mu_alpha = 4.0 / X[1]        
    
    return mu_gas, mu_e, mu_p, mu_alpha


#===================================================
#========== Main function
#===================================================
if __name__ == "__main__":

    print('========== Cluster global module ============')
    print('This gather modules dedicated to compute     ')
    print('global cluster quantities.                   ')
    print('                                             ')
    print('Here we present some example and checks.     ')    
    print('=============================================')
    print('')

    redshift = 0.0183
    M500     = 1e15
    c500     = 3.0
    cosmo    = astropy.cosmology.WMAP9

    #---------- Mass and radius
    R500 = Mdelta_to_Rdelta(M500, redshift, delta=500, cosmo=cosmo)
    M200 = Mdelta1_to_Mdelta2_NFW(M500, delta1=500, delta2=200, c1=c500, redshift=redshift, cosmo=cosmo)
    R200 = Mdelta_to_Rdelta(M200, redshift, delta=200, cosmo=cosmo)

    print(('redshift : '+str(redshift)))
    print(('M500     : '+str(M500)))
    print(('M200     : '+str(M200)))
    print(('R500     : '+str(R500)))
    print(('R200     : '+str(R200)))
    print('')

    #---------- Check NFW normalization
    r = np.logspace(0,4, 1000)
    Rs       = R500 / c500                                                     # kpc
    c200     = R200 / Rs
    rho01    = M500 / (4*np.pi * Rs**3) / (np.log(1 + c500) - c500/(1+c500))  # Msun/kpc^3
    rho02    = M200 / (4*np.pi * Rs**3) / (np.log(1 + c200) - c200/(1+c200))  # Msun/kpc^3

    plt.plot(r, rho01 * (r/Rs)**-1 * (1 + r/Rs)**-2, '-r', label='NFW normalized with input R500, c500')
    plt.plot(r, rho02 * (r/Rs)**-1 * (1 + r/Rs)**-2, '-.b', label='NFW normalized with recomputed R200, c200')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('radius (kpc)')
    plt.ylabel('density (M$_{\\odot}$ kpc$^{-3}$)')
    plt.legend()
    plt.show()

    #---------- Check concentration effects
    print('M200/M500 for c500=[1,2,3,4,5,6,7,8,9] : ')
    for i in range(9): print((Mdelta1_to_Mdelta2_NFW(M500, delta1=500, delta2=200, c1=i+1)/M500))
    print('R200/R500 for c500=[1,2,3,4,5,6,7,8,9] : ')
    for i in range(9):
        M200_c = Mdelta1_to_Mdelta2_NFW(M500, delta1=500, delta2=200, c1=i+1)
        R200_c = Mdelta_to_Rdelta(M200_c, redshift, delta=200, cosmo=cosmo)
        print((R200_c/R500))
    print('')

    #---------- Gas physics information
    Pnorm = gNFW_normalization(redshift, M500, cosmo=cosmo)
    print(('GNFW normalization : '+str(Pnorm)+' keV cm-3'))

    mu_gas, mu_e, mu_p, mu_alpha = mean_molecular_weight(Y=0.245, Z=0.0)
    print(('mu_gas   : '+str(mu_gas)))
    print(('mu_p     : '+str(mu_p)))
    print(('mu_e     : '+str(mu_e)))
    print(('mu_alpha : '+str(mu_alpha)))
    print(('n_p/n_e  : '+str(mu_e/mu_p)))

