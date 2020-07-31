"""
This file contains a library of useful tools.

"""
import astropy.units as u
import numpy as np
import scipy.interpolate as interpolate


#==================================================
# Check radius
#==================================================

def check_qarray(qarr, unit=None):
    """
    Make sure quantity array are arrays

    Parameters
    ----------
    - qarr (quantity): array or float, homogeneous to some unit

    Outputs
    ----------
    - qarr (quantity): quantity array

    """

    if unit is None:
        if type(qarr) == float or type(qarr) == np.float64:
            qarr = np.array([qarr])
            
    else:
        try:
            test = qarr.to(unit)
        except:
            raise TypeError("Unvalid unit for qarr")

        if type(qarr.to_value()) == float or type(qarr.to_value()) == np.float64:
            qarr = np.array([qarr.to_value()]) * qarr.unit

    return qarr


#==================================================
# Make 2 d grid from 2 1d arrays
#==================================================

def replicate_array(x, N, T=False):
    """
    Make a two dimension grid based on two 1 dimension arrays,
    such as energy and radius.

    Parameters
    ----------
    - x (quantity): array one
    - Nrep (int): number of time to replicate
    - T (bool): transpose or not

    Outputs
    ----------
    - x_grid (quantity): x1 replicated along one direction 
    as a 2d quantity array

    """
    
    if T:
        x_grid = (np.tile(x, [N,1])).T
    else:
        x_grid = (np.tile(x, [N,1]))

    return x_grid


#==================================================
# Def array based on point per decade, min and max
#==================================================

def sampling_array(xmin, xmax, NptPd=10, unit=False):
    """
    Make an array with a given number of point per decade
    from xmin to xmax

    Parameters
    ----------
    - xmin (quantity): min value of array
    - xmax (quantity): max value of array
    - NptPd (int): the number of point per decade

    Outputs
    ----------
    - array (quantity): the array

    """

    if unit:
        my_unit = xmin.unit
        array = np.logspace(np.log10(xmin.to_value(my_unit)),
                            np.log10(xmax.to_value(my_unit)),
                            int(NptPd*(np.log10(xmax.to_value(my_unit)/xmin.to_value(my_unit)))))*my_unit
    else:
        array = np.logspace(np.log10(xmin), np.log10(xmax), int(NptPd*(np.log10(xmax/xmin))))

    return array


#==================================================
# Integration loglog space with trapezoidale rule
#==================================================

def trapz_loglog(y, x, axis=-1, intervals=False):
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
    - trapz (float): Definite integral as approximated by trapezoidal rule in loglog space.
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


#==================================================
# Compute spectrum integration
#==================================================

def energy_integration(func, energy, Energy_density=False):
    """
    Integrate over the energy to get the profile, or the flux:
    \int_Emin^Emax (E) dN_dEdVdt(E,r) dE
    \int_Emin^Emax (E) dN_dEdt(E) dE
    
    Parameters
    ----------
    - dN_dEdVdt (2d or 1d array): Input array to integrate.
    - energy (array): energy variable to integrate over.
    
    Returns
    -------
    - dN_dVdt (array): integrated quantity

    """
    
    if Energy_density:
        if func.ndim == 1:
            I_func = trapz_loglog(energy*func, energy, axis=0, intervals=False)
        elif func.ndim == 2:
            I_func = trapz_loglog(np.vstack(energy.value)*energy.unit*func, energy, axis=0, intervals=False)
    else:        
        I_func = trapz_loglog(func, energy, axis=0, intervals=False)
        
    return I_func


#==================================================
# Compute spherical integration
#==================================================

def spherical_integration(func, radius):
    """
    Integrate over the spherical volume to get the spectrum:
    \int_Rmin^Rmax 4 pi r^2 dN_dEdVdt(E,r) dr
    
    Parameters
    ----------
    - dN_dEdVdt (1d/2d array): Input array to integrate.
    - radius (array): Radius variable to integrate over.
    
    Returns
    -------
    - dN_dEdt (array): integrated quantity

    """
    if func.ndim == 1:
        I = trapz_loglog(4*np.pi*radius**2*func, radius, axis=0, intervals=False)
    elif func.ndim == 2:
        I = trapz_loglog(4*np.pi*radius**2*func, radius, axis=1, intervals=False)
        
    return I


#==================================================
# Compute cylindrical integration
#==================================================

def cylindrical_integration(dN_dEdVdt, eng, r3d, r2d, los, Rtrunc=None):
    """
    Integrate over the spherical cylindrical volume to get the spectrum:
    \int_Rmin^Rmax 2 pi r dr \int_Rmin^Rmax dN_dEdVdt(E,r) dl
    
    Parameters
    ----------
    - dN_dEdVdt (2d array): a function of energy and radius.
    - eng (quantity array): energy
    - r2d (quantity array): projected radius
    - los (quantity array): line of sight (one side only)
    
    Returns
    -------
    - dN_dEdt (array): integrated quantity

    """

    # We need real 2 d array for los_integration_2dfunc, check this
    was2d = False
    if dN_dEdVdt.ndim == 2:
        if len(dN_dEdVdt[:,0]) == 1:
            dN_dEdVdt = dN_dEdVdt.flatten()
            was2d = True

    # Case of 2d array
    if dN_dEdVdt.ndim == 2:
        # First compute the los integral assuming a 2d function to get: f(E, r2d)
        dN_dEdVdt_proj = los_integration_2dfunc(dN_dEdVdt, eng, r3d, r2d, los)
        
        # In case of a truncation, some ringing can happen with interpolation, deal with this
        if Rtrunc is not None:
            rgrid = replicate_array(r2d, len(eng), T=False)
            dN_dEdVdt_proj[rgrid > Rtrunc] = 0
            
        # Then integrate over the surface
        dN_dEdt = trapz_loglog(2*np.pi*r2d*dN_dEdVdt_proj, r2d, axis=1, intervals=False)

    # Case of 1d array
    if dN_dEdVdt.ndim == 1:
        # First compute the los integral assuming a 1d function to get: f(r2d) at E_0
        dN_dEdVdt_proj = los_integration_1dfunc(dN_dEdVdt, r3d, r2d, los)

        # In case of a truncation, some ringing can happen with interpolation, deal with this
        if Rtrunc is not None:
            dN_dEdVdt_proj[r2d > Rtrunc] = 0
            
        # Then integrate over the surface
        dN_dEdt = trapz_loglog(2*np.pi*r2d*dN_dEdVdt_proj, r2d, axis=0, intervals=False)

        # Make it a 1 D array in case of dim = [1, Nrad]
        if was2d:
            dN_dEdt = np.array([dN_dEdt.value])*dN_dEdt.unit

    return dN_dEdt


#==================================================
# Compute l.o.s. integral for a 2d function
#==================================================

def los_integration_2dfunc(f_E_r, eng, r3d, r2d, los):
    """
    Compute the line of sight integral in the case of a 
    function with dependance on E and r:
    \int_Rmin^Rmax y(E,r) dl
    
    Parameters
    ----------
    - f_E_r (function): a function of energy and radius.
    In practice energy can be anything.
    - eng (array): energy
    - r2d (array): projected radius
    - los (array): line of sight (one side only)
    
    Returns
    -------
    - I_los (array): integrated quantity

    """

    # make sure unit are coherent since array manipulation does not handle units
    r2d = r2d.to(r3d.unit)
    los = los.to(r3d.unit)
    fEr_unit = f_E_r.unit
    f_E_r = f_E_r.to_value()
    
    # Get the number of elements
    Neng = len(eng)
    Nr2d = len(r2d)
    Nlos = len(los)

    # Compute 2d grids as Nr2d, Nlos to get the 3d radius array
    r2d_g2 = replicate_array(r2d, Nlos, T=True)
    los_g2 = replicate_array(los, Nr2d, T=False)
    r3d_g2 = np.sqrt(r2d_g2**2 + los_g2**2)        # in the same unit as r3d
    
    # Get a flat radius array and sort it
    r3d_g2_flat = np.ndarray.flatten(r3d_g2)
    r3d_g2_flat_sort = np.sort(r3d_g2_flat)
    index_sort = np.argsort(r3d_g2_flat)   # r3d_g2_flat[index_sort] gives r3d_g2_flat_sort
    index = np.argsort(index_sort)         # r3d_g2_flat_sort[index] gives r3d_g2_flat

    # Interpolated the function at the new position    
    itpl = interpolate.interp2d(r3d, eng, f_E_r, kind='cubic')
    f_E_r_g2_flat_sort = itpl(r3d_g2_flat_sort, eng)

    # Reshaping to make it as expected: unsorted and Neng x Nr2d x Nlos
    f_E_r_g2_flat = f_E_r_g2_flat_sort[:,index]
    f_E_r_g3 = np.reshape(f_E_r_g2_flat, (Neng, Nr2d, Nlos))
    
    # compute integral
    I_los = trapz_loglog(2*f_E_r_g3*fEr_unit, los, axis=2, intervals=False)
    
    return I_los


#==================================================
# Compute l.o.s. integral for a 2d function
#==================================================

def los_integration_1dfunc(f_r, r3d, r2d, los):
    """
    Compute the line of sight integral in the case of a 
    function with dependance on r:
    \int_Rmin^Rmax y(r) dl
    
    Parameters
    ----------
    - f_r (function): a function of radius.
    - r2d (array): projected radius
    - los (array): line of sight (one side only)
    
    Returns
    -------
    - I_los (array): integrated quantity

    """

    # make sure unit are coherent since array manipulation does not handle units
    r2d = r2d.to(r3d.unit)
    los = los.to(r3d.unit)
    fr_unit = f_r.unit
    f_r = f_r.to_value()
    
    # Get the number of elements
    Nr2d = len(r2d)
    Nlos = len(los)

    # Compute 2d grids as Nr2d, Nlos to get the 3d radius array
    r2d_g2 = replicate_array(r2d, Nlos, T=True)
    los_g2 = replicate_array(los, Nr2d, T=False)
    r3d_g2 = np.sqrt(r2d_g2**2 + los_g2**2)        # in the same unit as r3d
    
    # Get a flat radius array and sort it
    r3d_g2_flat = np.ndarray.flatten(r3d_g2)
    r3d_g2_flat_sort = np.sort(r3d_g2_flat)
    index_sort = np.argsort(r3d_g2_flat)   # r3d_g2_flat[index_sort] gives r3d_g2_flat_sort
    index = np.argsort(index_sort)         # r3d_g2_flat_sort[index] gives r3d_g2_flat

    # Interpolated the function at the new position    
    itpl = interpolate.interp1d(r3d, f_r, kind='cubic')
    f_r_g2_flat_sort = itpl(r3d_g2_flat_sort)

    # Reshaping to make it as expected
    f_r_g2_flat = f_r_g2_flat_sort[index]
    f_r_g2 = np.reshape(f_r_g2_flat, (Nr2d, Nlos))
    
    # compute integral
    I_los = trapz_loglog(2*f_r_g2*fr_unit, los, axis=1, intervals=False)
    
    return I_los
