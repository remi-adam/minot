import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
import warnings

#===================================================
#========== CREATE A MAP FINE FOR TEMPLATE FITTING
#===================================================
def map_to_gamma_template(in_file,
                          out_file=None,
                          no_neg=False,
                          smooth_arcmin=0.0,
                          rm_median=False,
                          flag_dist_arcmin=0.0,
                          normalize=True):
    """ 
    PURPOSE: This function extracts a map from a fits fil and modifies it 
    applying smoothing, baseline removal, flag of negative values, 
    and masking. The new image is then writtin in a new file.

    INPUT: - in_file (string): input file full name
           - out_file (string): (output file full name)
           - no_deg (bool): set negative values to zero (default is no)
           - smooth_arcmin (float): Gaussian smoothing FWHM, in arcmin, to apply
           - rm_median (bool): remove the median of the map (default is no)
           - flag_dist_arcmin (float): set to zero pixels beyond this limit, 
             from the center
           - normalize (bool): set this keyword to normalize the map to 1 (i.e.
             setting the integral sum(map)*reso^2 to 1)

    OUTPUT: - The new map is written in a new file
            - image (2d numpy array): the new map
            - header : the corresponding map header
    """

    #---------- Data extraction
    data = fits.open(in_file)[0]
    image = data.data
    header = data.header
    wcs = WCS(header)
    reso_x = abs(wcs.wcs.cdelt[0])
    reso_y = abs(wcs.wcs.cdelt[1])
    Npixx = image.shape[0]
    Npixy = image.shape[1]
    
    #---------- Data modification
    wnan = np.isnan(image)
    image[wnan] = 0.0
    winf = np.isinf(image)
    image[winf] = 0.0
    
    if smooth_arcmin >= 0:
        sigma_sm = smooth_arcmin/60.0/np.array([reso_x, reso_x])/(2*np.sqrt(2*np.log(2)))
        image = ndimage.gaussian_filter(image, sigma=sigma_sm)
        
    if rm_median:
        image = image - np.nanmedian(image)

    if no_neg:
        image[image < 0] = 0.0

    if flag_dist_arcmin > 0:
        ra_map, dec_map = get_radec_map(data.header)
        ra_ctr = np.mean(ra_map)
        dec_ctr = np.mean(dec_map)
        distmap = greatcircle(ra_map, dec_map, ra_ctr, dec_ctr)
        image[distmap > flag_dist_arcmin/60.0] = 0.0

    if normalize == True:
        norm = np.sum(image) * reso_x * reso_y * (np.pi/180.0)**2
        image = image/norm
        
    #---------- Write FITS
    if out_file is not None:
        hdu = fits.PrimaryHDU(header=header)
        hdu.data = image
        hdu.writeto(out_file, overwrite=True)
    
    return image, header

#===================================================
#========== Give the map normalization
#===================================================
def get_map_norm(image, header) :
    """
    Measure the normalization of a map

    Parameters
    ----------
    - image: input map
    - header: input header
    
    Outputs
    --------
    - norm ([image] * sr): the integral of the map over all solid angle
    """
    w = WCS(header)
    reso_x = np.abs(w.wcs.cdelt[0])
    reso_y = np.abs(w.wcs.cdelt[1])
    
    norm = np.sum(image) * reso_x * reso_y * (np.pi/180.0)**2

    return norm

#===================================================
#========== Build standard wcs and header
#===================================================
def define_std_header(ra_center, dec_center, FoV_x, FoV_y, reso, force_odd=True) :
    """
    Build a header and wcs object for a standard map

    Parameters
    ----------
    - ra_center (deg): coordinate of the R.A. reference center
    - dec_center (deg): coordinate of the Dec. reference center
    - FoV_x (deg): size of the map along x axis
    - FoV_y (deg): size of the map along y axis
    - reso (deg): the map resolution
    - force_odd (bool): force the number of pixels to be odd

    Outputs
    --------
    - header

    """
        
    Naxisx = int(FoV_x/reso)
    Naxisy = int(FoV_y/reso)

    # Makes it odd to have one pixel at the center
    if force_odd:
        if Naxisx/2.0 == int(Naxisx/2.0): Naxisx += 1
        if Naxisy/2.0 == int(Naxisy/2.0): Naxisy += 1

    data_tpl = np.zeros((Naxisy, Naxisx))
    
    w = WCS(naxis=2)
    w.wcs.crpix = (np.array([Naxisx, Naxisy])-1)/2+1
    w.wcs.cdelt = np.array([-reso, reso])
    w.wcs.crval = [ra_center, dec_center]
    w.wcs.latpole = 90.0
    w.wcs.lonpole = 180.0
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    header = w.to_header()
    
    hdu = fits.PrimaryHDU(header=header, data=data_tpl)
    header = hdu.header
    w = WCS(header)

    return header

#===================================================
#========== CREATE R.A. and Dec. maps from wcs
#===================================================
def get_radec_map(header):
    """
    Extract a RA and Dec map from a map header and a reference coordinate

    Parameters
    ----------
    - header: header associated to the map

    Outputs
    --------
    - ra_map (deg): map of R.A. values
    - dec_map (deg): map of Dec. values

    Example
    -------
    hdu = fits.open(file)[0]
    ra_map, dec_map = get_radec_map(hdu)

    """

    w = WCS(header)
    Naxis1 = header['NAXIS1']
    Naxis2 = header['NAXIS2']
    
    axis1 = np.arange(0, Naxis1)
    axis2 = np.arange(0, Naxis2)
    coord_x, coord_y = np.meshgrid(axis1, axis2, indexing='xy')
    world = w.wcs_pix2world(coord_x, coord_y, 0)
    
    ra_map = world[0]
    dec_map = world[1]
    
    return ra_map, dec_map

#===================================================
#========== Compute great circle angles on the sky
#===================================================
def greatcircle(lon, lat, lon_ref, lat_ref):
    """
    Compute distances between points on a sphere.

    Parameters
    ----------
    - lon (deg): longitude (should be np array)
    - lat (deg): latitude (should be np array)
    - lon_ref (deg): reference longitude
    - lat_ref (deg): reference latitude

    Outputs
    --------
    - angle (deg): np array containing the angular distance

    """
    
    arg1 = 180.0/np.pi*2
    arg2 = (np.sin((lat_ref-lat)*np.pi/180.0/2.0))**2
    arg3 = np.cos(lat*np.pi/180.0) * np.cos(lat_ref*np.pi/180.0)
    arg4 = (np.sin((lon_ref-lon)*np.pi/180.0/2.0))**2
    
    wbad1 = arg2 + arg3 * arg4 < 0
    if np.sum(wbad1) > 0: print('WARNING : '+str(np.sum(wbad1))+' Bad coord')
    
    angle = arg1 * np.arcsin(np.sqrt(arg2 + arg3 * arg4))
                             
    return angle

#===================================================
#========== Interpolate profile onto map
#===================================================
def profile2map(profile_y, profile_r, map_r):
    """
    Interpolated a profile onto a map

    Parameters
    ----------
    - profile_y: amplitude value of the profile at a given radius
    - profile_r (deg): radius corresponding to profile_y
    - map_r (deg): radius map in 2d

    Outputs
    --------
    - map_y (in units of profile_y): interpolated map

    """

    map_r_flat = np.reshape(map_r, map_r.shape[0]*map_r.shape[1])
    itpl = interpolate.interp1d(profile_r, profile_y, kind='cubic', fill_value='extrapolate')
    map_y_flat = itpl(map_r_flat)
    map_y = np.reshape(map_y_flat, (map_r.shape[0],map_r.shape[1]))
    
    return map_y

#===================================================
#========== Extract ROI from Healpix maps
#===================================================
def roi_extract_healpix(file_name, ra, dec, reso_deg, FoV_deg, save_file=None, visu=True, field=0):
    """
    Extract a sky patch on the sky given a healpix fullsky map

    Parameters
    ----------
    - file_name (str): Healpix fits file in galactic coordinates
    - ra (deg)  : the R.A. coordinate of the center
    - dec (deg) : the Dec. coordinate of the center
    - reso_deg (deg): the map resolution in degrees
    - FoV_deg (deg): the field of view of the extracted map as 2d list (x,y)
    - save_file (str): the name of the file where to save the map
    - visu (bool): visualize the map
    - field (int): the field where to extract the healpix map

    Outputs
    --------
    - image: the extracted map
    - head: the corresponding header
    
    """
    
    try:
        import healpy
    except:
        print("Healpy is not installed while it is requiered by roi_extract_healpix")

    #======== Preparation
    FoV_x = FoV_deg[0]
    FoV_y = FoV_deg[1]
    coord = SkyCoord(ra, dec, frame="icrs", unit="deg")
        
    #======== Map header and geometry
    head_roi = define_std_header(ra, dec, FoV_x, FoV_y, reso_deg) 
    
    #======== Read the healpix map
    image_hp, head_hp = healpy.fitsfunc.read_map(file_name, field=field, hdu=1, h=True, verbose=False)

    #======== Color map
    cmap = copy.copy(mpl.cm.get_cmap("viridis"))
    
    #======== Show the full sky map
    if visu:        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=mpl.cbook.mplDeprecation)
            healpy.visufunc.mollview(map=image_hp,
                                     xsize=800,
                                     min=None, max=None,
                                     cmap=cmap,
                                     notext=False,
                                     norm='hist',
                                     hold=False,
                                     return_projected_map=False)
            healpy.visufunc.projscatter(coord.galactic.l.value, coord.galactic.b.value,
                                        lonlat=True,
                                        marker='o', s=80, facecolors='white', edgecolors='black')
            healpy.graticule()
            
    #======== Extract the gnomview
    if visu:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=mpl.cbook.mplDeprecation)
            image_roi = healpy.visufunc.gnomview(map=image_hp,
                                                 coord=('G', 'C'),
                                                 rot=(coord.ra.value, coord.dec.value, 0.0),
                                                 xsize=head_roi['Naxis1'], ysize=head_roi['Naxis2'],
                                                 reso=60.0*reso_deg,
                                                 cmap=cmap,
                                                 norm='hist',
                                                 hold=False,
                                                 return_projected_map=True,
                                                 no_plot=False)
            healpy.graticule()
            
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=mpl.cbook.mplDeprecation)
            image_roi = healpy.visufunc.gnomview(map=image_hp,
                                                 coord=('G', 'C'),
                                                 rot=(coord.ra.value, coord.dec.value, 0.0),
                                                 xsize=head_roi['Naxis1'], ysize=head_roi['Naxis2'],
                                                 reso=60.0*reso_deg,
                                                 cmap=cmap,
                                                 hold=False,
                                                 return_projected_map=True,
                                                 no_plot=True)

    #======== Save the data
    if save_file != '' and save_file is not None :
        hdu = fits.PrimaryHDU(header=head_roi)
        hdu.data = image_roi.data
        hdu.writeto(save_file, overwrite=True)        

    #======== Print out the maps
    if visu:
        plt.show()        
        
    return image_roi, head_roi

#===================================================
#========== Extract ROI from Healpix maps
#===================================================
def get_healpix_dec_mask(dec_lim, nside):
    """
    Compute a Healpix binnary mask given a limit declinaison

    Parameters
    ----------
    - dec_lim (deg): the limit declinaison for the mask
    - nside : Healpix Nside argument

    Outputs
    --------
    - mask (int array): the binnary mask cut at dec_lim
    
    """

    try:
        import healpy
    except:
        print("Healpy is not installed while it is requiered by get_healpix_dec_mask")

    # Compute longitude latitude maps
    ipix = np.linspace(0, healpy.nside2npix(nside), healpy.nside2npix(nside), dtype=int)
    angle = healpy.pix2ang(nside, ipix, lonlat=False)
    maplon = angle[1] * 180.0/np.pi
    maplat = 90.0 - angle[0] * 180.0/np.pi

    # Get the Dec map
    mapcoord = SkyCoord(maplon, maplat, frame="galactic", unit="deg")
    mapdec = mapcoord.icrs.dec.value

    # Compute a mask
    wmask = mapdec < dec_lim
    mask = np.zeros(healpy.nside2npix(nside))
    mask[wmask] = 1

    return mask


#===================================================
#========== Compute the radial profile of a count map
#===================================================
def radial_profile_cts(image, center,
                       stddev=None,
                       header=None,
                       binsize=1.0,
                       stat='GAUSSIAN',
                       counts2brightness=False,
                       residual=True):
    """
    Compute the radial profile of an image in units proportional 
    to counts (e.g. counts per pixel)

    Parameters
    ----------
    - image (2D array) : input map
    - center (tupple) : coord along x and y. In case a header is given, these
    are R.A. and Dec. in degrees, otherwise this is in pixel.
    - stddev (2D array) : the standard deviation map. In case of Gaussian statistics,
    this is the sigma of the gaussian noise distribution in each pixel. In case of 
    Poisson statistics, we have stddev = sqrt(expected counts)
    - header (string) : header that contains the astrometry
    - binsize (float): the radial bin size, in degree if the header is provided and 
    in pixel unit otherwise.
    - stat (string): 'GAUSSIAN' (for surface brightness like images), 
    and 'POISSON' for counts
    - counts2brightness (bool): set to true if you want to normalized by the solid
    angle. This should be set to True for surface brightness like average
    - residual (bool): is the map a residual map? If yes, the data will be taken as 
    residual+model in poisson counts 

    Outputs
    --------
    - r_ctr (1D array): the center of the radial bins
    - p (1D array): the profile
    - err (1D array): the uncertainty
    
    """

    #----- Use constant weight if no stddev given
    if stddev is None:
        stddev = image*0+1.0
        print('!!! WARNING: The stddev is set to 1 for each pixels.')
        if stat == 'POISSON':
            print('             For POISSON statistics, stddev should be sqrt(expected_value).')
            print('             This affects only the uncertainty of the profile.')
        if stat == 'GAUSSIAN':
            print('             For GAUSSIAN statistics stddev is the noise standard deviation in each pixel.')
            print('             This affects the profile uncertainty and is used for weights')
            
    #------ Get the distance from the center map
    if header is None:
        y, x = np.indices((image.shape))
        dist_map = np.sqrt((x-center[0])**2+(y-center[1])**2)  # in pixels
        reso_x = 1.0 # pixel size in unit of pixel
        reso_y = 1.0
    else:
        ra_map, dec_map = get_radec_map(header)
        dist_map = greatcircle(ra_map, dec_map, center[0], center[1]) # in deg
        w = WCS(header)
        reso_x = np.abs(w.wcs.cdelt[0]) # pixel size in unit of deg
        reso_y = np.abs(w.wcs.cdelt[1])
    
    dist_max = np.max(dist_map)

    #----- Compute the binning
    Nbin = int(np.ceil(dist_max/binsize))
    r_in  = np.linspace(0, dist_max, Nbin+1)
    r_out = r_in + (np.roll(r_in, -1) - r_in)
    r_in  = r_in[0:-1]
    r_out = r_out[0:-1]

    #----- Compute the profile
    r_ctr = np.array([])
    p     = np.array([])
    err   = np.array([])
    for i in range(Nbin):
        # Get the map pixels in the bin
        w_bin_rad =  (dist_map < r_out[i]) * (dist_map >= r_in[i])
        w_bin_val = (stddev > 0) * (np.isnan(stddev) == False) * (np.isnan(image) == False)
        w_bin     = w_bin_rad * w_bin_val
        Npix_bin  = np.sum(w_bin)

        # Gaussian case (mean x Npix_bin == counts in bin)
        if stat == 'GAUSSIAN':
            if Npix_bin > 0:
                val     = Npix_bin*np.sum((image/stddev**2)[w_bin]) / np.sum((1.0/stddev**2)[w_bin])
                val_err = Npix_bin / np.sqrt(np.sum((1.0/stddev**2)[w_bin]))
            else:
                val     = np.nan
                val_err = np.nan
                
        # Poisson case (mean x Npix_bin == counts in bin)
        if stat == 'POISSON':
            if Npix_bin > 0:
                cts     = np.sum(image[w_bin])
                cts_exp = np.sum((stddev**2)[w_bin]) # stddev**2 == model for poisson
                if residual:
                    cts_dat = cts + cts_exp
                else:
                    cts_dat = cts + 0.0
                log_ratio = np.log(np.float64(cts_dat)/np.float64(cts_exp)) # avoid 0 when dat~exp
                if np.abs(log_ratio)==np.inf: log_ratio = 0
                sig     = np.sign(cts_dat-cts_exp)*np.sqrt(2*(cts_dat*log_ratio+cts_exp-cts_dat))
                val     = cts*1.0 # cts/pixel or cts/deg^2
                val_err = val/sig
            else:
                val     = 0.0 # cts/pixel or cts/deg^2
                val_err = 0.0

        # Convert counts to counts per deg2 if needed
        if counts2brightness:
            if Npix_bin > 0:
                val     /= (Npix_bin*reso_x*reso_y)
                val_err /= (Npix_bin*reso_x*reso_y)
            else:
                val     = np.nan
                val_err = np.nan

        # Append the bin values
        p       = np.append(p, val)
        err     = np.append(err, val_err)
        r_ctr   = np.append(r_ctr, (r_in[i] + r_out[i])/2.0)

    return r_ctr, p, err


#===================================================
#========== Compute the radial profile of a count map
#===================================================
def radial_profile_sb(image, center,
                      stddev=None,
                      header=None,
                      binsize=1.0):
    """
    Compute the radial profile of an image in units of brightness (e.g. Jy/sr)

    Parameters
    ----------
    - image (2D array) : input map
    - center (tupple) : coord along x and y. In case a header is given, these
    are R.A. and Dec. in degrees, otherwise this is in pixel.
    - stddev (2D array) : the standard deviation map. In case of Gaussian statistics,
    this is the sigma of the gaussian noise distribution in each pixel. In case of 
    Poisson statistics, we have stddev = sqrt(expected counts)
    - header (string) : header that contains the astrometry
    - binsize (float): the radial bin size, in degree if the header is provided and 
    in pixel unit otherwise.

    Outputs
    --------
    - r_ctr (1D array): the center of the radial bins
    - p (1D array): the profile
    - err (1D array): the uncertainty
    
    """

    #----- Use constant weight if no stddev given
    if stddev is None:
        stddev = image*0+1.0
        print('!!! WARNING: The stddev is set to 1 for each pixels.')
        print('             stddev is the noise standard deviation in each pixel.')
        print('             This affects the profile uncertainty and is used for weights')
        
    #------ Get the distance from the center map
    if header is None:
        y, x = np.indices((image.shape))
        dist_map = np.sqrt((x-center[0])**2+(y-center[1])**2)  # in pixels
        reso_x = 1.0 # pixel size in unit of pixel
        reso_y = 1.0
    else:
        ra_map, dec_map = get_radec_map(header)
        dist_map = greatcircle(ra_map, dec_map, center[0], center[1]) # in deg
        w = WCS(header)
        reso_x = np.abs(w.wcs.cdelt[0]) # pixel size in unit of deg
        reso_y = np.abs(w.wcs.cdelt[1])
    
    dist_max = np.max(dist_map)

    #----- Compute the binning
    Nbin = int(np.ceil(dist_max/binsize))
    r_in  = np.linspace(0, dist_max, Nbin+1)
    r_out = r_in + (np.roll(r_in, -1) - r_in)
    r_in  = r_in[0:-1]
    r_out = r_out[0:-1]

    #----- Compute the profile
    r_ctr = np.array([])
    p     = np.array([])
    err   = np.array([])
    for i in range(Nbin):
        # Get the map pixels in the bin
        w_bin_rad =  (dist_map < r_out[i]) * (dist_map >= r_in[i])
        w_bin_val = (stddev > 0) * (np.isnan(stddev) == False) * (np.isnan(image) == False)
        w_bin     = w_bin_rad * w_bin_val
        Npix_bin  = np.sum(w_bin)

        # Gaussian case (mean x Npix_bin == counts in bin)
        if Npix_bin > 0:
            val     = np.sum((image/stddev**2)[w_bin]) / np.sum((1.0/stddev**2)[w_bin])
            val_err = 1.0 / np.sqrt(np.sum((1.0/stddev**2)[w_bin]))
        else:
            val     = np.nan
            val_err = np.nan

        # Append the bin values
        p       = np.append(p, val)
        err     = np.append(err, val_err)
        r_ctr   = np.append(r_ctr, (r_in[i] + r_out[i])/2.0)

    return r_ctr, p, err
