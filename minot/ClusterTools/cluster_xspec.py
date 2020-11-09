"""
This script contains Xspec tools.
It requires to have xspec installed on your machine.

"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt

from minot.ClusterTools import map_tools


#==================================================
# Get the hydrogen column density at the clsuter position
#==================================================

def get_nH(filename, ra, dec, fov=1.0, reso=0.1, save_file=None, visu=False):
    """
    Extract the hydrogen column density.
        
    Parameters
    ----------
    - filename (str): full path to nH healpix map
    - ra (deg): R.A. of the field center
    - dec (deg): declinaition of the field center
    - fov (deg): the patch size in which to extract the rms
    - reso (deg): the map resolution to extract
    - save_file (str): full path to the file to save if needed
    - visu (bool): set to True to visualize the output plot

    Outputs
    ----------
    - nH (float): hydrogen column density (in unit of 10^22 cm-2)
    - nH_rms (float): hydrogen column density field rms (in unit of 10^22 cm-2)
    
    """

    if type(fov) == float:        
        fov_x = fov
        fov_y = fov
    elif type(fov) == int:
        fov_x = float(fov)
        fov_y = float(fov)
    elif type(fov) == list:
        if len(fov) == 2:
            fov_x = float(fov[0])
            fov_y = float(fov[1])
    elif type(fov) == np.ndarray:        
        if len(fov) == 2:
            fov_x = float(fov[0])
            fov_y = float(fov[1])
    else:
        raise TypeError("Problem with FoV")
        
    img, head = map_tools.roi_extract_healpix(filename, ra, dec, reso, [fov_x,fov_y], save_file=save_file, visu=visu)
    img1d = (img.flatten()*1e-22).data

    nH = np.median(img1d)
    nH_rms = np.std(img1d)
    
    if visu:
        plt.figure(1)
        plt.hist(img1d)
        plt.xlabel('$n_H$ (10$^{22}$ cm$^{-2}$)')
        plt.ylabel('Npix')
        plt.show()

        print(('----- nH = '+str(nH)+' +/- '+str(nH_rms)))

    return nH, nH_rms


#==================================================
# Read fluxes from an Xspec output
#==================================================

def get_xspec_flux(filename):
    """
    Read X-ray fluxes and counts from an Xspec output
        
    Parameters
    ----------
    - filename (str): full path to the file to read

    Outputs
    ----------
    - flux (float): the X-ray flux in erg/cm^2/s
    - Counts (float): the X-ray counts in ph/cm^2/s
    - rate (float): the X-ray rate in ph/s for a instrumental response file
    
    """
    
    # Read the file
    f = open(filename, "r")
    cont = f.read()

    # Define the position where to read the text
    p1 = (re.compile('Model Flux')).search(cont)
    p2 = (re.compile('photons \(')).search(cont)
    p3 = (re.compile('ergs/cm\^2/s')).search(cont)
    p4 = (re.compile('Model predicted rate:')).search(cont)
    p5 = (re.compile('XSPEC: quit')).search(cont)

    # Case of error
    if p1 == None or p2 == None or p3 == None:
        raise ValueError("No XSPEC output found in "+filename)

    # Find the text between these places
    counts = float(cont[p1.end():p2.start()])
    flux  = float(cont[p2.end():p3.start()])

    # Get the rate for the instrument if it was given
    if p4 is not None:
        rate   = float(cont[p4.end():p5.start()])
    else:
        rate = np.nan
    
    return flux, counts, rate


#==================================================
# Create model
#==================================================

def make_xspec_file(nH, Tgas, ab, redshift, emin, emax,
                    filename='./xspec_analysis.txt', model='APEC',
                    resp_file=None, data_file=None, app_nH_model=False):
    """
    Create a file ready for Xspec, with normalization (see MEKAL or APEC in xspec) set to 1.
        
    Parameters
    ----------
    - nH (float): hydrogen column density (10^22 cm-2)
    - Tgas (float): plasma temperature (keV)
    - ab (float): abundances (in unit of Z0)
    - redshift (float) cluster redshift
    - emin (float): minimal band energy (keV) at observer
    - emax (float): maximal band energy (keV) at observer
    - filename (str): file name to be created
    - model (str): which model to use
    - resp_file (str): full path to the response file of e.g., ROSAT PSPC
    - data_file (str): full path to any data spectrum file needed for template in xspec 
    (see https://heasarc.gsfc.nasa.gov/FTP/rosat/doc/xselect_guide/xselect_guide_v1.1.1/xselect_ftools.pdf,
    section 5)
    - app_nH_model (bool): apply nH absorbtion to the initial model without instrumental effects

    Outputs
    ----------

    """

    #---------- Check inputs
    if model != 'APEC' and model != 'MEKAL':
        raise ValueError("Available models are APEC or MEKAL.")

    if resp_file == None and data_file is not None:
        print('!!! WARNING: both data_file and resp_file should be given to account for instrument response')
    if resp_file is not None and data_file == None:
        print('!!! WARNING: both data_file and resp_file should be given to account for instrument response')
    
    #---------- Decide if nH should be applyied also to the initial model
    nH_ini = 0.0
    if app_nH_model:
        nH_ini = nH

    #---------- Create the file
    with open(filename, 'w') as txtfile:
        # Get model without galactic absorb or instrumental effect
        if model == 'APEC':
            txtfile.write('model phabs(apec) & ' +str(nH_ini)+',-1 & '+str(Tgas)+',-1 & '+str(ab)+',-1 & '+str(redshift)+',-1 & 1,-1 \n')
        if model == 'MEKAL':
            txtfile.write('model phabs(mekal) & '+str(nH_ini)+',-1 & '+str(Tgas)+',-1 & / & '+str(ab)+',-1 & '+str(redshift)+',-1 & 0 & 1,-1 \n')
        txtfile.write('flux '+str(float(emin))+' '+str(float(emax))+' \n')

        # Get the rate given a response file
        if resp_file is not None and data_file is not None:
            txtfile.write('set xs_return_result 1 \n')
            txtfile.write('query yes \n')
            txtfile.write('data '+data_file+' \n')
            txtfile.write('response '+resp_file+' \n')
            txtfile.write('ignore **-'+str(float(emin))+' '+str(float(emax))+'-** \n')
            if model == 'APEC':
                txtfile.write('model phabs(apec) & ' +str(nH)+',-1 & '+str(Tgas)+',-1 & '+str(ab)+',-1 & '+str(redshift)+',-1 & 1,-1 \n')
            if model == 'MEKAL':
                txtfile.write('model phabs(mekal) & '+str(nH)+',-1 & '+str(Tgas)+',-1 & / & '+str(ab)+',-1 & '+str(redshift)+',-1 & 0 & 1,-1 \n')
            txtfile.write('show rate \n')


#==================================================
# Create model
#==================================================

def run_xspec(nH, Tgas, ab, redshift, emin, emax,
              file_ana='./xspec_analysis.txt', file_out='./xspec_analysis_output.txt',
              model='APEC', resp_file=None, data_file=None, app_nH_model=False,
              cleanup=True):
    """
    Run Xspec with a given model, but normalization set to 1.
        
    Parameters
    ----------
    - nH (float): hydrogen column density (10^22 cm-2)
    - Tgas (float): plasma temperature (keV)
    - ab (float): abundances (in unit of Z0)
    - redshift (float) cluster redshift
    - emin (float): minimal band energy (keV)
    - emax (float): maximal band energy (keV)
    - filename (str): file name to be created
    - model (str): which model to use
    - resp_file (str): full path to the response file of e.g., ROSAT PSPC
    - data_file (str): full path to any data spectrum file needed for template in xspec 
    (see https://heasarc.gsfc.nasa.gov/FTP/rosat/doc/xselect_guide/xselect_guide_v1.1.1/xselect_ftools.pdf,
    section 5)
    - app_nH_model (bool): apply nH absorbtion to the initial model without instrumental effects

    Outputs
    ----------
    - flux (float): the X-ray flux in erg/cm^2/s
    - Counts (float): the X-ray counts/cm^2/s
    - rate (float): the X-ray rate in ph/s for a instrumental response file

    """

    make_xspec_file(nH, Tgas, ab, redshift, emin, emax, filename=file_ana, model=model,
                    resp_file=resp_file, data_file=data_file, app_nH_model=app_nH_model)
    
    if os.path.isfile(file_out): os.remove(file_out)
    runXSPEC = 'xspec '+file_ana+' > '+file_out
    os.system(runXSPEC)

    flux, counts, rate = get_xspec_flux(file_out)

    if cleanup:
        if os.path.isfile(file_ana): os.remove(file_ana)
        if os.path.isfile(file_out): os.remove(file_out)

    return flux, counts, rate


#==================================================
# Compute X-ray spectrum
#==================================================

def xray_spectrum(nH, Tgas, ab, redshift,
                  emin=0.5, emax=10.0, nbin=100,
                  file_ana='./xspec_analysis.txt', file_out='./xspec_analysis_output.txt',
                  model='APEC', resp_file=None, data_file=None, cleanup=True,
                  logspace=True):
    """
    Compute an xray spectrum given a model.
        
    Parameters
    ----------
    - nH (float): hydrogen column density (10^22 cm-2)
    - Tgas (float): plasma temperature (keV)
    - ab (float): abundances (in unit of Z0)
    - redshift (float) cluster redshift
    - emin (float): minimal band energy (keV), for observer
    - emax (float): maximal band energy (keV), for observer
    - nbin (int): number of bins
    - file_ana (str): xspec analysis file name to be created
    - file_out (str): xspec analysis output file name to be created
    - model (str): which model to use
    - resp_file (str): full path to the response file of e.g., ROSAT PSPC
    - data_file (str): full path to any data spectrum file needed for template in xspec 
    (see https://heasarc.gsfc.nasa.gov/FTP/rosat/doc/xselect_guide/xselect_guide_v1.1.1/xselect_ftools.pdf,
    section 5)
    - cleanup (bool): clean the temporary file
    - logspace (bool): scale for the energy binning

    Outputs
    ----------
    - dNdE (np.ndarray): photon counts per unit energy (counts/cm^2/s/keV)
    - dSxdE (np.ndarray): photon energy per unit energy (erg/cm^2/s/keV)
    - ectr (np.ndarray): energy bin center (keV)
    - epot (np.ndarray): energy bin values (keV)

    """
    
    if logspace:
        epot = np.logspace(np.log10(emin), np.log10(emax), nbin+1)
    else:
        epot = np.linspace(emin, emax, nbin+1)
                
    ectr = ((epot+np.roll(epot,-1))/2.0)[0:-1]
    esiz = (np.roll(epot,-1) - epot)[0:-1]
    
    dNph = np.zeros(len(ectr))
    dSB  = np.zeros(len(ectr))
    dR   = np.zeros(len(ectr))
    for i in range(len(ectr)):
        flux, counts, rate = run_xspec(nH, Tgas, ab, redshift, epot[i], epot[i+1],
                                       file_ana=file_ana, file_out=file_out,
                                       model=model, resp_file=resp_file, data_file=data_file,
                                       app_nH_model=True, cleanup=cleanup)
        dNph[i] = counts / esiz[i]
        dSB[i]  = flux /   esiz[i]
        dR[i]   = rate /   esiz[i]

    return dSB, dNph, dR, ectr, epot


#==================================================
# Compute X-ray spectrum
#==================================================

def make_xspec_table(output_file, nH, ab, redshift, emin, emax,
                     Tmin=0.1, Tmax=50, nbin=100,
                     file_ana='./xspec_analysis.txt', file_out='./xspec_analysis_output.txt',
                     model='APEC', resp_file=None, data_file=None, app_nH_model=False, cleanup=True,
                     logspace=True):

    """
    Create a flux/counts versus Tgas table.
        
    Parameters
    ----------
    - output_file (str): full path to the output saved table
    - nH (float): hydrogen column density (10^22 cm-2)
    - ab (float): abundances (in unit of Z0)
    - redshift (float) cluster redshift
    - emin (float): minimal band energy (keV)
    - emax (float): maximal band energy (keV)
    - Tmin (float): minimal plasma temperature (keV)
    - Tmax (float): maximal plasma temperature (keV)
    - nbin (int): number of bins
    - file_ana (str): xspec analysis file name to be created
    - file_out (str): xspec analysis output file name to be created
    - model (str): which model to use
    - resp_file (str): full path to the response file of e.g., ROSAT PSPC
    - data_file (str): full path to any data spectrum file needed for template in xspec 
    (see https://heasarc.gsfc.nasa.gov/FTP/rosat/doc/xselect_guide/xselect_guide_v1.1.1/xselect_ftools.pdf,
    section 5)
    - app_nH_model (bool): apply nH absorbtion to the initial model without instrumental effects
    - cleanup (bool): clean the temporary file
    - logspace (bool): scale for the energy binning

    Outputs
    ----------
    - Table created

    """
    
    if logspace:
        Tvec = np.logspace(np.log10(Tmin), np.log10(Tmax), nbin)
    else:
        Tvec = np.linspace(Tmin, Tmax, nbin)
        
    dNdtdS = np.zeros(nbin) # ph/s/cm^2  cm^5
    dEdtdS = np.zeros(nbin) # erg/s/cm^2 cm^5
    dNdt   = np.zeros(nbin) # ph/s       cm^5
    
    # Loop over all temperature
    for i in range(nbin):
        flux, counts, rate = run_xspec(nH, Tvec[i], ab, redshift, emin, emax,
                                       file_ana=file_ana, file_out=file_out,
                                       model=model, resp_file=resp_file, data_file=data_file,
                                       app_nH_model=app_nH_model, cleanup=cleanup)
        dNdtdS[i] = counts
        dEdtdS[i] = flux
        dNdt[i]   = rate

    # saving
    sfile = open(output_file, 'w')
    sfile.writelines(['#nH = '+str(nH)+' 10^22 cm^2 ; abundance = '+str(ab)+
                      ' Zsun ; redshift = '+str(redshift)+
                      ' ; energy=['+str(emin)+','+str(emax)+']'+
                      ' ; model = '+model+' ; Absorb in counts & flux = '+str(app_nH_model)+'\n'])
    sfile.writelines(['#Output normalized to 10^{-14} / (4\pi D_A^2 (1+z)^2) \int n_e n_p dV [cm^-5]\n'])
    sfile.writelines(['#T (keV)         ', '   ', 'Counts (ph/s/cm2)', '   ', 'Flux (erg/s/cm2) ', '   ', 'Rate (ph/s)'+'\n'])
    for il in range(nbin):
        sfile.writelines([('{:.5e}').format(Tvec[il]),    '         ',
                          ('{:.5e}').format(dNdtdS[il]),  '         ',
                          ('{:.5e}').format(dEdtdS[il]),  '         ',
                          ('{:.5e}').format(dNdt[il]) +'\n'])
    sfile.close()

    
#==================================================
# Make data.sp and responses
#==================================================

def how_to_handle_xspec_data():

    """
    Provide information on how to deal with XSPEC data
        
    Parameters
    ----------

    Outputs
    ----------

    """

    print('In order to compute the expected count rate when running cluster_xspec modules,')
    print('one needs to include the instrumental response. This can be done as follows:')
    print('')
    print('Step 1')
    print('Download data from the archive, see https://heasarc.gsfc.nasa.gov/docs/rosat/rhp_archive.html')
    print('For cluster, see also ftp://legacy.gsfc.nasa.gov/rosat/data/pspc/processed_data/clusters_of_galaxies.html')
    print('')
    print('Step 2.')
    print('Download the response matrices')
    print('see https://heasarc.gsfc.nasa.gov/docs/rosat/pspc_matrices.html')
    print('')
    print('Step 3')
    print('run XSELECT: $ xselect')
    print('set the data directory (where you downloaded the repository), e.g. > set datadir /Users/adam/Downloads/rp800005n00')
    print('read the events (*_bas.fits file), e.g. > read events rp800005n00_bas.fits')
    print('extract the spectrum > extract spectrum')
    print('save the spectrum as, e.g. > save spectrum source.sp')
    print('')
    print('Step 4')
    print('Run cluster_xspec modules with source.sp as the data file and the response matrices as response file')
    print('')
    
