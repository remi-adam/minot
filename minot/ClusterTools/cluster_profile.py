""" 
This script gather several functions which are related to the profile
of galaxy clusters.
"""

import numpy as np
from scipy.special import gamma
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

from minot.ClusterTools import map_tools

#===================================================
#========== Beta model
#===================================================
def beta_model(r3d_kpc, n0, r_c, beta):
    """
    Compute a beta model

    Parameters
    ----------
    - r3d_kpc: array of radius in kpc
    - r_c : core radius parameter
    - n_0  : normalization
    - beta : slope of the profile

    Outputs
    --------
    - beta model profile as a function of the input radius vector

    """
    
    return n0 * (1 + (r3d_kpc / r_c)**2)**(-3.0*beta/2.0)

#===================================================
#========== Beta model
#===================================================
def beta_model_derivative(r3d_kpc, n0, r_c, beta):
    """
    Compute the derivative of a beta model dn/dr

    Parameters
    ----------
    - r3d_kpc: array of radius in kpc
    - r_c : core radius parameter
    - n_0  : normalization
    - beta : slope of the profile

    Outputs
    --------
    - beta model derivative profile as a function of the input radius vector

    """

    return -3.0*n0*beta*r3d_kpc*(1+(r3d_kpc/r_c)**2)**(-3.0*beta/2.0-1.0)/r_c**2
    
#===================================================
#========== Simplified Vikhlinin model
#===================================================
def svm_model(r3d_kpc, n0, r_c, beta, r_s=1.0, gamma=3.0, epsilon=0.0, alpha=0.0):
    """
    Compute a Simplified Vikhlinin model

    Parameters
    ----------
    - r3d_kpc: array of radius in kpc
    - r_c : core radius parameter
    - n_0  : normalization
    - beta : slope of the profile
    - r_s  : characteristic radius parameter (kpc)
    - gamma : slope of the profile
    - epsilon : slope of the profile

    Outputs
    --------
    - SVM model profile as a function of the input radius vector

    """

    term1 = n0 * (1 + (r3d_kpc / r_c)**2)**(-3.0*beta/2.0)
    term2 = (r3d_kpc / r_c)**(-alpha/2.0)
    term3 = (1 + (r3d_kpc / r_s)**gamma)**(-epsilon/2.0/gamma)
    
    return term1 * term2 * term3

#===================================================
#========== Simplified Vikhlinin model
#===================================================
def svm_model_derivative(r3d_kpc, n0, r_c, beta, r_s=1.0, gamma=3.0, epsilon=0.0, alpha=0.0):
    """
    Compute the derivative of the Simplified Vikhlinin model

    Parameters
    ----------
    - r3d_kpc: array of radius in kpc
    - r_c : core radius parameter
    - n_0  : normalization
    - beta : slope of the profile
    - r_s  : characteristic radius parameter (kpc)
    - gamma : slope of the profile
    - epsilon : slope of the profile

    Outputs
    --------
    - SVM derivative model profile as a function of the input radius vector

    """
    
    t1 = n0 / (r3d_kpc*(r_c**2 + r3d_kpc**2))
    t2 = (r3d_kpc / r_c)**(-alpha/2.0)
    t3 = (1 + (r3d_kpc / r_c)**2)**(-3.0*beta/2.0)
    t4 = (1 + (r3d_kpc/r_s)**gamma)**(-epsilon/2.0/gamma - 1.0)
    t51 = -alpha * (r_c**2 + r3d_kpc**2) * ((r3d_kpc/r_s)**gamma + 1) / 2.0
    t52 = -3*beta*r3d_kpc**2*((r3d_kpc/r_s)**gamma + 1)
    t53 = -epsilon*(r_c**2 + r3d_kpc**2) / 2.0 * (r3d_kpc/r_s)**gamma

    return t1*t2*t3*t4*(t51+t52+t53)

#===================================================
#========== NFW model
#===================================================
def NFW_model(r3d_kpc, norm, rs):
    """
    Compute a NFW model

    Parameters
    ----------
    - r3d_kpc (kpc): array of radius
    - norm : the normalization 
    - rs (kpc): characteristic radius parameter

    Outputs
    --------
    - NFW model profile as a function of the input radius vector

    """

    # rho_0 = delta_c * rho_crit
    # rs = R_delta / c_delta
    # delta_c = Delta/3 * c^3 (log(1+c) - c/(1+c))
    # rho_0 = Mdelta / (4 pi rs^3) / (log(1+c) - c/(1+c))
    
    return norm / (r3d_kpc / rs) / (1 + (r3d_kpc / rs))**2


#===================================================
#========== NFW model derivative
#===================================================
def NFW_model_derivative(r3d_kpc, norm, rs):
    """
    Compute the derivative of the NFW model (dP/dr)

    Parameters
    ----------
    - r3d_kpc (kpc): array of radius
    - norm : the normalization 
    - rs (kpc): characteristic radius parameter

    Outputs
    --------
    - NFW derivative model profile as a function of the input radius vector

    """

    t1 = -norm * (r3d_kpc / rs)**-1 * (1 + (r3d_kpc / rs))**-3
    t2 = (3*r3d_kpc/rs + 1) / r3d_kpc
    
    return t1 * t2


#===================================================
#========== gNFW model
#===================================================
def gNFW_model(r3d_kpc, P0, r_p, slope_a=1.33, slope_b=4.13, slope_c=0.31):
    """
    Compute a gNFW model

    Parameters
    ----------
    - r3d_kpc (kpc): array of radius
    - P0 : normalization
    - r_p (kpc): characteristic radius parameter
    - sope_a : intermediate slope parameter
    - sope_b : outer slope parameter
    - sope_c : inner slope parameter

    Outputs
    --------
    - gNFW model profile as a function of the input radius vector

    """
        
    return P0 * (r3d_kpc / r_p)**(-slope_c) * (1 + (r3d_kpc / r_p)**(slope_a))**((slope_c-slope_b)/slope_a)

#===================================================
#========== gNFW model
#===================================================
def gNFW_model_derivative(r3d_kpc, P0, r_p, slope_a=1.33, slope_b=4.13, slope_c=0.31):
    """
    Compute the derivative of the gNFW model (dP/dr)

    Parameters
    ----------
    - r3d_kpc (kpc): array of radius
    - P0 : normalization
    - r_p (kpc): characteristic radius parameter
    - sope_a : intermediate slope parameter
    - sope_b : outer slope parameter
    - sope_c : inner slope parameter

    Outputs
    --------
    - gNFW derivative model profile as a function of the input radius vector

    """

    t1 =  -P0 * (r3d_kpc / r_p)**(-slope_c)
    t2 = (1 + (r3d_kpc / r_p)**(slope_a))**((slope_c-slope_b-slope_a)/slope_a)
    t3 = (slope_b*(r3d_kpc/r_p)**slope_a + slope_c) / r3d_kpc
    
    return t1 * t2 * t3

#===================================================
#========== Project beta model
#===================================================
def proj_beta_model(r2d_kpc, n0, r_c, beta):
    """
    Compute a projected beta model:
    P(R) = \int n_e dl at given R

    Parameters
    ----------
    - r2d_kpc: array of projected radius at which to compute integration
    - n0  : normalization
    - r_c : core radius parameter
    - beta : slope of the profile

    Outputs
    --------
    - The projected profile in units of kpc times original profile

    """
    
    return np.sqrt(np.pi) * n0 * r_c * gamma(1.5*beta - 0.5) / gamma(1.5*beta) * (1 + (r2d_kpc/r_c)**2)**(0.5-1.5*beta)

#===================================================
#========== Integrated any model
#===================================================
def proj_any_model_naive(r3d, p3d, Rpmax=6000.0, Rpstep=10.0):
    """
    Compute the projection of any given profile in a 
    naive way (simple steps and trapezoidal integration).
    P(R) = \int p(r) dl for values of R
    
    Parameters
    ----------
    - r3d (kpc): input radius array
    - p3d (arb): input profile array
    - Rpmax (kpc): projected range up to which we compute the profile
    - Rpstep (kpc): size of the steps along projection direction

    Outputs
    --------
    - r2d (kpc): output projected radius array
    - p2d (arb kpc): output projected profile

    """

    rproj_Nstep = int(Rpmax/Rpstep)
    r2d = np.linspace(Rpstep, Rpmax, num=rproj_Nstep)
    p2d = np.empty(rproj_Nstep)
    
    for istep in range(0,rproj_Nstep):
        w = r3d > r2d[istep]
        r3d_w = r3d[w]
        p3d_w = p3d[w]
        integ = 2.0 * p3d_w * r3d_w / np.sqrt((r3d_w**2.0 - r2d[istep]**2.0))
        p2d[istep] = np.trapz(integ, x=r3d_w)
        
    return r2d, p2d

#===================================================
#========== Integrated any model
#===================================================
def proj_any_model(r3d, p3d, Npt=100, Rmax=5000.0, Rpmax=6000.0, Rpstep=10.0, Rp_input=None):
    """
    Compute the projection of any given profile. This is done by 
    creating a lin-log mapping of the distance from the center and summing 
    the pixels directly. It works very well in terms of speed and is very accurate.
    P(R) = \int p(r) dl for values of R
    
    Parameters
    ----------
    - r3d (kpc): input radius array
    - p3d (arb): input profile array
    - Npt : number of point in the integration line of sight
    - Rmax (kpc): limit radius for integration
    - Rpmax (kpc): projected range up to which we compute the profile
    - Rpstep (kpc): size of the steps along projection direction
    - Rp_input (kpc): input vector for the projected radius

    Outputs
    --------
    - r2d (kpc): output projected radius array
    - p2d (arb kpc): output projected profile

    """

    #---------- Line-of-Sight grid
    los_bin = np.logspace(0, np.log10(Rmax), num=Npt)
    los_bin = np.concatenate((np.array([0]), los_bin), axis=None)
    
    los_step = los_bin - np.roll(los_bin, 1)              # Steps
    los_cbin = (los_bin + np.roll(los_bin, 1))/2.0        # Center of bins
    los_step = los_step[1:]                               # remove first bin which does not exist
    los_cbin = los_cbin[1:]                               # remove first bin which does not exist
    
    #---------- Projected radius grid
    if Rp_input is not None :
        Rproj = Rp_input
        rproj_Nstep = len(Rproj)
    else:
        rproj_Nstep = int(Rpmax/Rpstep)                       # Number of bins along L.o.S    
        Rproj = np.linspace(Rpstep, Rpmax, num=rproj_Nstep)   # Vector going up to rproj_out with rp_step steps
    
    #---------- Compute the radius grid
    los_cbin_map = np.tile(los_cbin, [rproj_Nstep,1])     # Line of sight bin central value map
    los_step_map = np.tile(los_step, [rproj_Nstep,1])     # Line of sight bin size map
    Rproj_map    = np.tile(Rproj, [Npt,1]).T              # Projected radius map
    r_map = np.sqrt(Rproj_map**2 + los_cbin_map**2)       # Physical radius from the center map

    #---------- Interpolate the profile to the grid
    P_Rl = map_tools.profile2map(p3d, r3d, r_map)
        
    #---------- Integration
    integrant = P_Rl * los_step_map
    lim_out = (r_map > Rmax)
    integrant[lim_out] = 0
    
    Pproj = 2*np.sum(integrant, axis=1)
        
    return Rproj, Pproj

#===================================================
#========== Volume integrate of beta model
#===================================================
def get_volume_beta_model(Rmax_kpc, n0, r_c, beta):
    """
    Compute the volume integrated beta model

    Parameters
    ----------
    - Rmax_kpc (kpc): radius at which to stop integration
    - n0  : normalization
    - r_c : core radius parameter
    - beta : slope of the profile

    Outputs
    --------
    - The integrated profile in units of kpc^3 times the original profile

    """
    
    # Defines a function which returns the integrand
    def get_volume_beta_model_integrand(r3d_kpc, n0, r_c, beta):    
        return 4.0*np.pi * r3d_kpc**2 * n0 * (1 + (r3d_kpc / r_c)**2)**(-3*beta/2.0)

    # Integrate the integrand
    output = integrate.quad(get_volume_beta_model_integrand, 0.0, Rmax_kpc, args=(n0, r_c, beta))[0]
    
    return output
    
#===================================================
#========== Volume any profile
#===================================================
def get_volume_any_model(r3d, p3d, Rmax_kpc, Npt=1000):
    """
    Compute the volume integrated profile:
    \int_0^Rmax 4pi r^2 p(r) dr

    Parameters
    ----------
    - Rmax_kpc (kpc): radius at which to stop integration
    - r3d (kpc) : input radius in kpc
    - p3d : input profile

    Outputs
    --------
    - The integrated profile in units of kpc^3 times the original profile

    Note
    --------
    The error can be large if the input profile is not well sampled. Use ~1000 
    points in r3d/p3d for safety, and well sampled until Rmax. See also 
    define_safe_radius_array() for this.

    """

    if Rmax_kpc > 1.001*np.amax(r3d):
        print('!!!!!!!!!! WARNING: you try to integrate beyond function limits !!!!!!!!!!')

    # interpolate to the location of interest
    r3d_i = np.logspace(np.log10(np.amin(r3d)), np.log10(Rmax_kpc), num=Npt)
    r3d_i = np.concatenate((np.array([0]), r3d_i), axis=None)

    itpl = interpolate.interp1d(r3d, p3d, kind='cubic', fill_value='extrapolate')
    p3d_i = itpl(r3d_i)

    # integrate
    output = np.trapz(4.0*np.pi * r3d_i**2 * p3d_i, r3d_i)

    return output

#===================================================
#========== Surface any profile
#===================================================
def get_surface_any_model(r2d, p2d, Rmax_kpc, Npt=1000):
    """
    Compute the surface integrated profile:
    \int_0^Rmax 2pi r p(r) dr

    Parameters
    ----------
    - Rmax_kpc (kpc): radius at which to stop integration
    - r2d (kpc) : input radius in kpc
    - p2d : input profile

    Outputs
    --------
    - The integrated profile in units of kpc^2 times the original profile

    Note
    --------
    
    """

    if Rmax_kpc > 1.001*np.amax(r2d):        
        print('!!!!!!!!!! WARNING: you try to integrate beyond function limits !!!!!!!!!!')
    
    # interpolate to the location of interest
    r2d_i = np.logspace(np.log10(np.amin(r2d)), np.log10(Rmax_kpc), num=Npt)
    r2d_i = np.concatenate((np.array([0]), r2d_i), axis=None)

    itpl = interpolate.interp1d(r2d, p2d, kind='cubic', fill_value='extrapolate')
    p2d_i = itpl(r2d_i)

    # integrate
    output = np.trapz(2.0*np.pi * r2d_i * p2d_i, r2d_i)

    return output

#===================================================
#========== Define radius vector
#===================================================
def define_safe_radius_array(r_in, Rmin=1.0, Rmax=None, Nptmin=1000):
    """
    Defines a radius that safely sample the profiles for integration, based 
    on the radius at which we want to get the samples

    Parameters
    ----------
    - r_in (kpc): input sampling radius
    - Rmin (kpc) : minimal for the sampling
    - Nptmin : minimal number of points in the sampling

    Outputs
    --------
    - r_out (kpc): radius used to sample the profile we want to integrate

    """

    # array case
    if len(r_in) > 1:
        logmax = np.amax(np.log10(r_in))
        logmin = np.amin(np.log10(r_in))
        Npt_per_decade = len(r_in) / (logmax - logmin)

        # We want at least to cover one decade
        if (logmax - logmin) < 1:
            logmin = logmax - 1
        
        # maximum radius remains the same, minimum radius should go to 1 kpc at least
        if Rmax is not None :
            if logmax < np.log10(Rmax):
                logmax_value = np.log10(Rmax)
            else:
                logmax_value = logmax
        else:
            logmax_value = logmax
            
        if logmin > np.log10(Rmin):
            logmin_value = np.log10(Rmin)
        else:
            logmin_value = logmin    

        # keep the same number of point per decade, with at leat Nptmin points
        Npt = int(Npt_per_decade * (logmax_value - logmin_value))
        if Npt < Nptmin: Npt = Nptmin

    # case a scalar is given
    else:
        r_in = float(r_in)
        Npt = Nptmin
        
        if Rmax is not None :
            if r_in < Rmax:
                logmax_value = np.log10(Rmax)
            else:
                logmax_value = np.log10(r_in)
        else:
            logmax_value = np.log10(r_in)

        if r_in/10.0 > Rmin:
            logmin_value = np.amin(np.log10(Rmin))
        else:
            logmin_value = np.log10(r_in/10.0)

    # Defines the radius
    r_out = np.logspace(logmin_value, logmax_value, Npt)
    
    return r_out

#===================================================
#========== Main function
#===================================================
if __name__ == "__main__":

    print('========== Cluster profile module ===========')
    print('This gather modules dedicated to compute     ')
    print('profile related quantities.                  ')
    print('                                             ')
    print('Here we present some example and checks.     ')    
    print('=============================================')

    # Cluster properties
    r3d = np.logspace(0,5, 100)    # radius (kpc)
    r_c = 200.0
    beta = 0.75
    n0 = 1.0

    # Get profiles in different ways
    n_r = beta_model(r3d, n0, r_c, beta)

    R2d, n_R1 = proj_any_model(r3d, n_r, Npt=1000, Rmax=10000.0, Rpmax=5000.0, Rpstep=10.0)
    n_R2      = proj_beta_model(R2d, n0, r_c, beta)

    vol1 = np.zeros(len(r3d))
    vol2 = np.zeros(len(r3d))
    for i in range(len(r3d)):
        vol1[i] = get_volume_beta_model(r3d[i], n0, r_c, beta)
        vol2[i] = get_volume_any_model(r3d, n_r, r3d[i], Npt=10000)
        
    # Plot the projected profile
    fig1 = plt.figure(1)
    plt.loglog(R2d, n_R1, label='Numerical projection')
    plt.loglog(R2d, n_R2, label='Analytical projection')
    plt.xlabel('Projected radius (kpc)')
    plt.ylabel('Projected profile')
    plt.title('Projected profile')
    plt.legend(loc='upper right')

    # Plot the relative error
    fig2 = plt.figure(2)
    plt.loglog(R2d, np.abs((n_R1 - n_R2)/n_R2) * 100)
    plt.xlabel('Projected radius (kpc)')
    plt.ylabel('Relative error (%)')
    plt.title('Error on projected profile')

    # Plot the volume integral  
    fig3 = plt.figure(3)
    plt.loglog(r3d, vol1, label='beta model integration')
    plt.loglog(r3d, vol2, label='general integration')
    plt.xlabel('radius (kpc)')
    plt.ylabel('Volume (<r)')
    plt.title('Volume integration')

    # Plot the relative error  
    fig3 = plt.figure(4)
    plt.loglog(r3d, np.abs((vol1-vol2)/vol1)*100)
    plt.xlabel('radius (kpc)')
    plt.ylabel('Relative error (%)')
    plt.title('Volume integration error')

    plt.show()

