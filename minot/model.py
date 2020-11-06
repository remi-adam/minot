"""
This file contains the Cluster class. It is dedicated to the construction of a 
Cluster object, definined by its physical properties and with  associated methods
to compute derived properties or observables. It focuses on the thermal and non-thermal 
component of the clusters ICM.

"""

#==================================================
# Requested imports
#==================================================

import numpy as np
import astropy.units as u
from astropy.io import fits
import astropy.cosmology
from astropy.coordinates import SkyCoord
from astropy import constants as const
from astropy.wcs import WCS

from minot              import model_title
from minot              import model_tools
from minot.model_admin  import Admin
from minot.model_modpar import Modpar
from minot.model_phys   import Physics
from minot.model_obs    import Observables
from minot.model_plots  import Plots
from minot.ClusterTools import cluster_global
from minot.ClusterTools import cluster_spectra


#==================================================
# Cluster class
#==================================================

class Cluster(Admin, Modpar, Physics, Observables, Plots):
    """ Cluster class. 
    This class defines a cluster object. In addition to basic properties such as 
    mass and redshift, it includes the physical properties (e.g. pressure profile, 
    cosmic ray spectrum) from which derived properties can be obtained (e.g. 
    hydrostatic mass profile) as well as observables.

    To do
    -----
    - Implement all functions in automatic plots
    - Make sure Epmin and Eemin are fine wrt integration of distribution
    - Check the effect of E{e/p}min on gamma ray interaction (i.e. v~c approx)
        
    Attributes
    ----------  
    - silent (bool): print information if False, or not otherwise.
    - output_dir (str): directory where to output data files and plots.

    - cosmology (astropy.cosmology): background cosmological model. Can only be set
    when creating the Cluster object.

    - name (str): the name of the cluster
    - coord (SkyCoord object): the coordinate of the cluster.
    - redshift (float): redshift of the cluster center. Changing the redshift 
    on the fly propagate to cluster properties.
    - D_ang (quantity): can be access but not set directly. The redshift+cosmo
    defines this.
    - D_lum (quantity) : can be access but not set directly. The redshift+cosmo
    defines this.
    - M500 (quantity) : the mass enclosed within R500.
    - R500 (quantity) : the radius in which the density is 500 times the critical 
    density at the cluster redshift
    - theta500 (quantity): the angle corresponding to R500.

    - R_truncation (quantity): the radius at which the cluster stops (similar as virial radius)
    - theta_truncation (quantity): the angle corresponding to R_truncation.

    - helium_mass_fraction (float): the helium mass fraction of the gas (==Yp~0.25 in BBN)
    - metallicity_sol (float): the metallicity (default is Zprotosun == 0.0153)
    - abundance (float): the abundance (default is 0.3) in unit of Zprotosun

    - EBL_model (str): the EBL model to use for gamma rays

    - Rmin (quantity): the minimum radius used to define integration arrays
    - hse_bias (float): the hydrostatic mass bias, as Mtrue = (1-b) Mhse
    - X_crp_E (dict): the cosmic ray proton to thermal energy and the radius used for normalization
    - X_cre1_E (dict): the primary cosmic ray electron to thermal energy and the radius used for normalization
    - Epmin (quantity): the minimal energy of protons (default is the threshold energy for 
    pi0 production)
    - Epmax (quantity): the maximal energy of protons (default is 10 PeV)
    - Eemin (quantity): the minimal energy of primary electrons (default is electron rest mass)
    - Eemax (quantity): the maximal energy of primary electrons (default is 10 PeV) 
    - pp_interaction_model (str) : model for particle physics parametrisation of pp 
    interactions. Available are 'Pythia8', 'SIBYLL', 'QGSJET', 'Geant4'.
    - cre1_loss_model (str): what kind of loss to apply on the primary CR distribution. If 
    None, the distribution is expected to be already dN/dEdV, else it is the injection rate 
    dN/dEdVdt

    - pressure_gas_model (dict): the model used for the thermal gas electron pressure 
    profile. It contains the name of the model and the associated model parameters. 
    - density_gas_model (dict): the model used for the thermal gas electron density 
    profile. It contains the name of the model and the associated model parameters. 
    - density_crp_model (dict): the definition of the cosmic ray proton radial shape
    - density_cre1_model (dict): the definition of the cosmic ray primary electron radial shape
    - magfield_model (dict): the definition of the magnetic field profile.
    - spectrum_crp_model (dict): the definition of the cosmic ray proton energy shape
    - spectrum_cre1_model (dict): the definition of the cosmic ray primary electron energy shape
    - density_cre1_model ; temporary
    - Npt_per_decade_integ (int): the number of point per decade used in integrations
    - map_coord (SkyCoord object): the map center coordinates.
    - map_reso (quantity): the map pixel size, homogeneous to degrees.
    - map_fov (list of quantity):  the map field of view as [FoV_x, FoV_y], homogeneous to deg.
    - map_header (standard header): this allows the user to provide a header directly.
    In this case, the map coordinates, field of view and resolution will be extracted 
    from the header and the projection can be arbitrary. If the header is not provided,
    then the projection will be standard RA-DEC tan projection.

    Methods
    ----------  
    Methods are split in the respective files: 
    - model_admin.py
    - model_modpar.py
    - model_phys.py
    - model_obs.py
    - model_plots.py

    """

    #==================================================
    # Initialize the cluster object
    #==================================================

    def __init__(self,
                 name='Cluster',
                 RA=0.0*u.deg, Dec=0.0*u.deg,
                 redshift=0.01,
                 M500=1e15*u.Unit('Msun'),
                 cosmology=astropy.cosmology.Planck15,
                 silent=False,
                 output_dir='./minot_output',
    ):
        """
        Initialize the cluster object. Several attributes of the class cannot 
        be defined externally because of intrications between parameters. For 
        instance, the cosmology cannot be changed on the fly because this would 
        mess up the internal consistency.
        
        Parameters
        ----------
        - name (str): cluster name 
        - RA, Dec (quantity): coordinates or the cluster in equatorial frame
        - redshift (float) : the cluster center cosmological redshift
        - M500 (quantity): the cluster mass 
        - cosmology (astropy.cosmology): the name of the cosmology to use.
        - silent (bool): set to true in order not to print informations when running 
        - output_dir (str): where to save outputs
        
        """
        
        #---------- Print the code header at launch
        if not silent:
            model_title.show()
        
        #---------- Admin
        self._silent     = silent
        self._output_dir = output_dir

        #---------- Check that the cosmology is indeed a cosmology object
        if hasattr(cosmology, 'h') and hasattr(cosmology, 'Om0'):
            self._cosmo = cosmology
        else:
            raise TypeError("Input cosmology must be an instance of astropy.cosmology")
        
        #---------- Global properties
        self._name     = name
        self._coord    = SkyCoord(RA, Dec, frame="icrs")
        self._redshift = redshift
        self._D_ang    = self._cosmo.angular_diameter_distance(self._redshift)
        self._D_lum    = self._cosmo.luminosity_distance(self._redshift)
        self._M500     = M500
        self._R500     = cluster_global.Mdelta_to_Rdelta(self._M500.to_value('Msun'),
                                                         self._redshift, delta=500, cosmo=self._cosmo)*u.kpc
        self._theta500 = ((self._R500 / self._D_ang).to('') * u.rad).to('deg')

        #---------- Cluster boundery
        self._R_truncation     = 3*self._R500
        self._theta_truncation = ((self._R_truncation / self._D_ang).to('') * u.rad).to('deg')

        #---------- ICM composition (default: protosolar from Lodders et al 2009: arxiv0901.1149)
        self._helium_mass_fraction = 0.2735
        self._metallicity_sol = 0.0153
        self._abundance = 0.3

        #---------- Extragalactic background light absorbtion
        self._EBL_model = 'dominguez'
        
        #---------- Physical properties
        self._Rmin = 1.0*u.kpc
        self._hse_bias = 0.2
        self._X_crp_E = {'X':0.01, 'R_norm':self._R500}
        self._X_cre1_E = {'X':0.01, 'R_norm': self._R500}
        self._Epmin = cluster_spectra.pp_pion_kinematic_energy_threshold() * u.GeV
        self._Epmax = 10.0 * u.PeV
        self._Eemin = (const.m_e *const.c**2).to('GeV')
        self._Eemax = 10.0 * u.PeV
        self._pp_interaction_model = 'Pythia8'
        self._cre1_loss_model = 'None'

        # Initialize the profile model (not useful but for clarity of variables)
        self._pressure_gas_model = 1
        self._density_gas_model  = 1
        self._density_crp_model  = 1
        self._density_cre1_model = 1
        self._magfield_model     = 1
        # Set default model using UPP + isoThermal + isobaric
        self.set_pressure_gas_gNFW_param(pressure_model='P13UPP')
        self.set_density_gas_isoT_param(10.0*u.keV)
        self.set_density_crp_isobaric_scal_param(scal=1.0)
        self.set_density_cre1_isobaric_scal_param(scal=1.0)
        self.set_magfield_isobaric_scal_param(Bnorm=10*u.uG, scal=0.5)
        # Cosmic ray protons
        self._spectrum_crp_model = {'name': 'PowerLaw',
                                    'PivotEnergy': 1.0*u.TeV,
                                    'Index': 2.5}
        # Cosmic ray primary electrons
        self._spectrum_cre1_model = {'name': 'PowerLaw',
                                     'PivotEnergy': 1.0*u.TeV,
                                     'Index': 3.0}
        
        #---------- Sampling
        self._Npt_per_decade_integ = 30
        self._map_coord  = SkyCoord(RA, Dec, frame="icrs")
        self._map_reso   = 0.01*u.deg
        self._map_fov    = [5.0, 5.0]*u.deg
        self._map_header = None
        
    #==================================================
    # Get the hidden variable
    #==================================================

    #========== Admin
    @property
    def silent(self):
        return self._silent

    @property
    def output_dir(self):
        return self._output_dir
    
    #========== Cosmology
    @property
    def cosmo(self):
        return self._cosmo

    #========== Global properties
    @property
    def name(self):
        return self._name
    
    @property
    def coord(self):
        return self._coord
    
    @property
    def redshift(self):
        return self._redshift

    @property
    def D_ang(self):
        return self._D_ang
    
    @property
    def D_lum(self):
        return self._D_lum
    
    @property
    def M500(self):
        return self._M500

    @property
    def R500(self):
        return self._R500

    @property
    def theta500(self):
        return self._theta500

    #========== Cluster boundary
    @property
    def R_truncation(self):
        return self._R_truncation
    
    @property
    def theta_truncation(self):
        return self._theta_truncation

    #========== ICM composition    
    @property
    def helium_mass_fraction(self):
        return self._helium_mass_fraction

    @property
    def abundance(self):
        return self._abundance

    @property
    def metallicity_sol(self):
        return self._metallicity_sol

    #========== EBL model
    @property
    def EBL_model(self):
        return self._EBL_model
    
    #========== ICM physics
    @property
    def Rmin(self):
        return self._Rmin
    
    @property
    def hse_bias(self):
        return self._hse_bias
    
    @property
    def X_crp_E(self):
        return self._X_crp_E

    @property
    def X_cre1_E(self):
        return self._X_cre1_E

    @property
    def Epmin(self):
        return self._Epmin

    @property
    def Epmax(self):
        return self._Epmax

    @property
    def Eemin(self):
        return self._Eemin

    @property
    def Eemax(self):
        return self._Eemax

    @property
    def pp_interaction_model(self):
        return self._pp_interaction_model
    
    @property
    def pressure_gas_model(self):
        return self._pressure_gas_model
    
    @property
    def density_gas_model(self):
        return self._density_gas_model

    @property
    def density_crp_model(self):
        return self._density_crp_model

    @property
    def density_cre1_model(self):
        return self._density_cre1_model

    @property
    def magfield_model(self):
        return self._magfield_model
    
    @property
    def spectrum_crp_model(self):
        return self._spectrum_crp_model

    @property
    def spectrum_cre1_model(self):
        return self._spectrum_cre1_model

    @property
    def cre1_loss_model(self):
        return self._cre1_loss_model

    #========== Maps parameters
    @property
    def Npt_per_decade_integ(self):
        return self._Npt_per_decade_integ

    @property
    def map_coord(self):
        return self._map_coord

    @property
    def map_reso(self):
        return self._map_reso

    @property
    def map_fov(self):
        return self._map_fov

    @property
    def map_header(self):
        return self._map_header

    #==================================================
    # Defines how the user can pass arguments and interconnections
    #==================================================

    #========== Admin
    @silent.setter
    def silent(self, value):
        # Check value and set
        if type(value) == bool:
            self._silent = value
        else:
            raise TypeError("The silent parameter should be a boolean.")

        # Information
        if not self._silent: print("Setting silent value")

    @output_dir.setter
    def output_dir(self, value):
        # Check value and set
        if type(value) == str:
            self._output_dir = value
        else:
            raise TypeError("The output_dir should be a string.")
        
        # Information
        if not self._silent: print("Setting output_dir value")

    #========== Cosmology
    @cosmo.setter
    def cosmo(self, value):
        message = ("The cosmology can only be set when defining the cluster object, "
                   "as clust = Cluster(cosmology=astropy.cosmology.YourCosmology).  "
                   "Doing nothing.                                                  ")
        if not self._silent: print(message)
    
    #========== Global properties
    @name.setter
    def name(self, value):
        # Check value and set
        if type(value) == str:
            self._name = value
        else:
            raise TypeError("The name should be a string.")

        # Information
        if not self._silent: print("Setting name value")

    @coord.setter
    def coord(self, value):
        # Case value is a SkyCoord object, nothing to be done
        if type(value) == astropy.coordinates.sky_coordinate.SkyCoord:
            self._coord = value

        # Case value is standard coordinates
        elif type(value) == dict:
            
            # It is not possible to have both RA-Dec and Glat-Glon, or just RA and not Dec, etc
            cond1 = 'RA'  in list(value.keys()) and 'Glat' in list(value.keys())
            cond2 = 'RA'  in list(value.keys()) and 'Glon' in list(value.keys())
            cond3 = 'Dec' in list(value.keys()) and 'Glat' in list(value.keys())
            cond4 = 'Dec' in list(value.keys()) and 'Glon' in list(value.keys())
            if cond1 or cond2 or cond3 or cond4:
                raise TypeError("The coordinates can be a coord object, or a {'RA','Dec'} or {'Glon', 'Glat'} dictionary.")
            
            # Case where RA-Dec is used
            if 'RA' in list(value.keys()) and 'Dec' in list(value.keys()):
                self._coord = SkyCoord(value['RA'], value['Dec'], frame="icrs")

            # Case where Glon-Glat is used
            elif 'Glon' in list(value.keys()) and 'Glat' in list(value.keys()):
                self._coord = SkyCoord(value['Glon'], value['Glat'], frame="galactic")

            # Otherwise, not appropriate value
            else:
                err_message = ("The coordinates can be a coord object, "
                               "a {'RA','Dec'} dictionary, or a {'Glon', 'Glat'} dictionary.")
                raise TypeError(err_message)

        # Case value is not accepted
        else:
            raise TypeError("The coordinates can be a coord object, a {'RA','Dec'} dictionary, or a {'Glon', 'Glat'} dictionary.")

        # Information
        if not self._silent: print("Setting coord value")

    @redshift.setter
    def redshift(self, value):
        # check type
        if type(value) != float and type(value) != int and type(value) != np.float64:
            raise TypeError("The redshift should be a int or a float.")
        
        # value check
        if value < 0 :
            raise ValueError("The redshift should be larger or equal to 0.")

        # Setting parameters
        self._redshift = value
        self._D_ang = self._cosmo.angular_diameter_distance(self._redshift)
        self._D_lum = self._cosmo.luminosity_distance(self._redshift)
        self._R500  = cluster_global.Mdelta_to_Rdelta(self._M500.to_value('Msun'),
                                                      self._redshift, delta=500, cosmo=self._cosmo)*u.kpc            
        self._theta500 = ((self._R500 / self._D_ang).to('') * u.rad).to('deg')
        self._theta_truncation = ((self._R_truncation / self._D_ang).to('') * u.rad).to('deg')
        
        # Information
        if not self._silent: print("Setting redshift value")
        if not self._silent: print("Setting: D_ang, D_lum, R500, theta500, theta_truncation ; Fixing: cosmo.")
        
    @D_ang.setter
    def D_ang(self, value):
        if not self._silent: print("The angular diameter distance cannot be set directly, the redshift has to be used instead.")
        if not self._silent: print("Doing nothing.                                                                            ")
        
    @D_lum.setter
    def D_lum(self, value):
        if not self._silent: print("The luminosity distance cannot be set directly, the redshift has to be used instead.")
        if not self._silent: print("Doing nothing.                                                                      ")
        
    @M500.setter
    def M500(self, value):
        # Type check
        try:
            test = value.to('Msun')
        except:
            raise TypeError("The mass M500 should be a quantity homogeneous to Msun.")

        # Value check
        if value <= 0 :
            raise ValueError("Mass M500 should be larger than 0")

        # Setting parameters
        self._M500 = value
        self._R500 = cluster_global.Mdelta_to_Rdelta(self._M500.to_value('Msun'),
                                                     self._redshift, delta=500, cosmo=self._cosmo)*u.kpc
        self._theta500 = ((self._R500 / self._D_ang).to('') * u.rad).to('deg')
        
        # Information
        if not self._silent: print("Setting M500 value")
        if not self._silent: print("Setting: R500, theta500 ; Fixing: redshift, cosmo, D_ang")
        
    @R500.setter
    def R500(self, value):
        # Check type
        try:
            test = value.to('kpc')
        except:
            raise TypeError("The radius R500 should be a quantity homogeneous to kpc.")

        # check value
        if value < 0 :
            raise ValueError("Radius R500 should be larger than 0")

        # Setting parameter
        self._R500 = value
        self._theta500 = ((self._R500 / self._D_ang).to('') * u.rad).to('deg')
        self._M500 = cluster_global.Rdelta_to_Mdelta(self._R500.to_value('kpc'),
                                                     self._redshift, delta=500, cosmo=self._cosmo)*u.Msun
        
        # Information
        if not self._silent: print("Setting R500 value")
        if not self._silent: print("Setting: theta500, M500 ; Fixing: redshift, cosmo, D_ang")
        
    @theta500.setter
    def theta500(self, value):
        # Check type
        try:
            test = value.to('deg')
        except:
            raise TypeError("The angle theta500 should be a quantity homogeneous to deg.")
        
        # Check value
        if value <= 0 :
            raise ValueError("Angle theta500 should be larger than 0")

        # Setting parameters
        self._theta500 = value
        self._R500 = value.to_value('rad')*self._D_ang
        self._M500 = cluster_global.Rdelta_to_Mdelta(self._R500.to_value('kpc'),
                                                     self._redshift, delta=500, cosmo=self._cosmo)*u.Msun
        
        # Information
        if not self._silent: print("Setting theta500 value")
        if not self._silent: print("Setting: R500, M500 ; Fixing: redshift, cosmo, D_ang")

    #========== Cluster boundary
    @R_truncation.setter
    def R_truncation(self, value):
        # Check type
        try:
            test = value.to('kpc')
        except:
            raise TypeError("The radius R_truncation should be a quantity homogeneous to kpc.")

        # check value
        if value <= self._R500 :
            raise ValueError("Radius R_truncation should be larger than R500 for internal consistency.")

        # Set parameters
        self._R_truncation = value
        self._theta_truncation = ((self._R_truncation / self._D_ang).to('') * u.rad).to('deg')
        
        # Information
        if not self._silent: print("Setting R_truncation value")
        if not self._silent: print("Setting: theta_truncation ; Fixing: D_ang")
        
    @theta_truncation.setter
    def theta_truncation(self, value):
        # Check type
        try:
            test = value.to('deg')
        except:
            raise TypeError("The angle theta_truncation should be a quantity homogeneous to deg.")
        
        # check value
        if value <= self._theta500 :
            raise ValueError("Angle theta_truncation should be larger than theta500 for internal consistency.")

        # Set parameters
        self._theta_truncation = value
        self._R_truncation = value.to_value('rad') * self._D_ang
        
        # Information
        if not self._silent: print("Setting theta_truncation value")
        if not self._silent: print("Setting: R_truncation ; Fixing: D_ang")
        
    #========== ICM composition
    @Rmin.setter
    def Rmin(self, value):
        # Check type
        try:
            test = value.to('kpc')
        except:
            raise TypeError("The radius Rmin should be a quantity homogeneous to kpc.")
        
        # Set parameters
        self._Rmin = value
        
        # Information
        if not self._silent: print("Setting Rmin value")
    
    @helium_mass_fraction.setter
    def helium_mass_fraction(self, value):
        # Check type
        if type(value) != float and type(value) != int and type(value) != np.float64:
            raise TypeError("The helium mass fraction should be a float or int")

        # Check value
        if value > 1.0 or value < 0.0:
            raise ValueError("The helium mass fraction should be between 0 and 1")

        # Set parameters
        self._helium_mass_fraction = value
        
        # Information
        if not self._silent: print("Setting helium mass fraction value")
        
    @metallicity_sol.setter
    def metallicity_sol(self, value):
        # Check type
        if type(value) != float and type(value) != int and type(value) != np.float64:
            raise TypeError("The metallicity should be a float")
        
        # Check value
        if value < 0.0:
            raise ValueError("The metallicity should be >= 0")

        # Set parameters
        self._metallicity_sol = value
        
        # Information
        if not self._silent: print("Setting metallicity value")

    @abundance.setter
    def abundance(self, value):
        # Check type
        if type(value) != float and type(value) != int and type(value) != np.float64:
            raise TypeError("The abundance should be a float")

        # Check value
        if value < 0.0:
            raise ValueError("The abundance should be >= 0")

        # Set parameters
        self._abundance = value
        
        # Information
        if not self._silent: print("Setting abundance value")
        
    #========== EBL model
    @EBL_model.setter
    def EBL_model(self, value):
        # Check type
        if type(value) != str:
            raise TypeError("The EBL model should be a string")

        # Check value
        ebllist = ['none', 'franceschini', 'kneiske', 'finke',
                   'dominguez', 'dominguez-upper', 'dominguez-lower',
                   'gilmore', 'gilmore-fixed']
        if not value in ebllist:
            print('EBL available models are:')
            print(ebllist)
            raise ValueError("This EBL model is not available")
        
        # Setting parameters
        self._EBL_model = value
        
        # Information
        if not self._silent: print("Setting EBL_model value")

    #========== Thermal gas physics        
    @hse_bias.setter
    def hse_bias(self, value):
        # Check type
        if type(value) != float and type(value) != int and type(value) != np.float64:
            raise TypeError("The hydrostatic mass bias should be a float or int")

        # Set parameter
        self._hse_bias = value

        # Information
        if not self._silent: print("Setting hydrostatic mass bias value")

    @X_crp_E.setter
    def X_crp_E(self, value):
        # Check type and content
        if type(value) != dict :
            raise TypeError("The CRp/thermal energy should be a dictionary as {'X':CR/th fraction, 'R_norm':enclosed radius}.")

        if 'X' in list(value.keys()) and 'R_norm' in list(value.keys()):
            # Check units and value
            try:
                test = value['R_norm'].to('kpc')
            except:
                raise TypeError("R_norm should be homogeneous to kpc")
            
            if value['X'] < 0:
                print(('value  is '+str(value['X'])))
                raise ValueError("The cosmic ray to thermal pressure ratio X should be >= 0")
            
            if value['R_norm'].to_value('kpc') <= 0:
                raise ValueError("The enclosed radius should be > 0")
            
            # Implement
            self._X_crp_E = {'X':value['X'], 'R_norm':value['R_norm'].to('kpc')}
            
        else:
            raise TypeError("The cosmic/thermal energy should be a dictionary as {'X':CR/th fraction, 'R_norm':enclosed radius}.")
        
        # Information
        if not self._silent: print("Setting cosmic ray to thermal pressure ratio value")

    @Epmin.setter
    def Epmin(self, value):
        # Check type
        try:
            test = value.to('GeV')
        except:
            raise TypeError("The minimal proton energy sould be a quantity homogeneous to GeV.")

        # Value check
        if value <= 0 :
            raise ValueError("Energy Epmin should be larger than 0")
        
        # Setting parameters
        self._Epmin = value
        
        # Information
        if not self._silent: print("Setting Epmin value")
        
    @Epmax.setter
    def Epmax(self, value):
        # Check type
        try:
            test = value.to('GeV')
        except:
            raise TypeError("The maximal proton energy sould be a quantity homogeneous to GeV.")

        # Value check
        if value <= 0 :
            raise ValueError("Energy Epmax should be larger than 0")
        
        # Setting parameters
        self._Epmax = value
        
        # Information
        if not self._silent: print("Setting Epmax value")

    @X_cre1_E.setter
    def X_cre1_E(self, value):
        # Check type and content
        if type(value) != dict :
            raise TypeError("The CRe/thermal energy should be a dictionary as {'X':CRe/th fraction, 'R_norm':enclosed radius}.")

        if 'X' in list(value.keys()) and 'R_norm' in list(value.keys()):
            # Check units and value
            try:
                test = value['R_norm'].to('kpc')
            except:
                raise TypeError("R_norm should be homogeneous to kpc")
            
            if value['X'] < 0:
                raise ValueError("The cosmic ray to thermal pressure ratio X should be >= 0")
            
            if value['R_norm'].to_value('kpc') <= 0:
                raise ValueError("The enclosed radius should be > 0")
            
            # Implement
            self._X_cre1_E = {'X':value['X'], 'R_norm':value['R_norm'].to('kpc')}
            
        else:
            raise TypeError("The cosmic/thermal energy should be a dictionary as {'X':CRe/th fraction, 'R_norm':enclosed radius}.")
        
        # Information
        if not self._silent: print("Setting cosmic ray to thermal pressure ratio value")

    @Eemin.setter
    def Eemin(self, value):
        # Check type
        try:
            test = value.to('GeV')
        except:
            raise TypeError("The minimal electron energy sould be a quantity homogeneous to GeV.")

        # Value check
        if value <= 0 :
            raise ValueError("Energy Eemin should be larger than 0")
        
        # Setting parameters
        self._Eemin = value
        
        # Information
        if not self._silent: print("Setting Eemin value")
        
    @Eemax.setter
    def Eemax(self, value):
        # Check type
        try:
            test = value.to('GeV')
        except:
            raise TypeError("The maximal electron energy sould be a quantity homogeneous to GeV.")

        # Value check
        if value <= 0 :
            raise ValueError("Energy Eemax should be larger than 0")
        
        # Setting parameters
        self._Eemax = value
        
        # Information
        if not self._silent: print("Setting Eemax value")

    @pp_interaction_model.setter
    def pp_interaction_model(self, value):
        # Check type
        if type(value) != str:
            raise TypeError("The pp interaction model should be a string")

        # Check value
        pp_llist = ['Pythia8', 'SIBYLL', 'QGSJET', 'Geant4']
        if not value in pp_llist:
            print('pp model available models are:')
            print(pp_llist)
            raise ValueError("This pp interaction model is not available")
        
        # Setting parameters
        self._pp_interaction_model = value
        
        # Information
        if not self._silent: print("Setting pp_interaction_model value")
        
    @pressure_gas_model.setter
    def pressure_gas_model(self, value):
        # check type
        if type(value) != dict :
            raise TypeError("The pressure gas model should be a dictionary containing the name key and relevant parameters")
        
        # Check the input parameters and use it
        Ppar = self._validate_profile_model_parameters(value, 'keV cm-3')
        self._pressure_gas_model = Ppar
        
        # Information
        if not self._silent: print("Setting pressure_gas_model value")
        if not self._silent: print("Fixing: R500")

    @density_gas_model.setter
    def density_gas_model(self, value):
        # check type
        if type(value) != dict :
            raise TypeError("The density gas model should be a dictionary containing the name key and relevant parameters")
        
        # Continue if ok
        Ppar = self._validate_profile_model_parameters(value, 'cm-3')
        self._density_gas_model = Ppar
        
        # Information
        if not self._silent: print("Setting density_gas_model value")
        if not self._silent: print("Fixing: R500")
        
    @density_crp_model.setter
    def density_crp_model(self, value):
        # check type
        if type(value) != dict :
            raise TypeError("The density CRp model should be a dictionary containing the name key and relevant parameters")
        
        # Continue if ok
        Ppar = self._validate_profile_model_parameters(value, '')
        self._density_crp_model = Ppar
        
        # Information
        if not self._silent: print("Setting density_crp_model value")
        if not self._silent: print("Fixing: R500")

    @density_cre1_model.setter
    def density_cre1_model(self, value):
        # check type
        if type(value) != dict :
            raise TypeError("The density CRe1 model should be a dictionary containing the name key and relevant parameters")
        
        # Continue if ok
        Ppar = self._validate_profile_model_parameters(value, '')
        self._density_cre1_model = Ppar
        
        # Information
        if not self._silent: print("Setting density_cre1_model value")
        if not self._silent: print("Fixing: R500")

    @magfield_model.setter
    def magfield_model(self, value):
        # check type
        if type(value) != dict :
            raise TypeError("The magnetic field model should be a dictionary containing the name key and relevant parameters")
        
        # Continue if ok
        if not self._silent: print(value)
        Ppar = self._validate_profile_model_parameters(value, 'uG')
        self._magfield_model = Ppar
        
        # Information
        if not self._silent: print("Setting magfield_model value")
        if not self._silent: print("Fixing: R500")

    @spectrum_crp_model.setter
    def spectrum_crp_model(self, value):
        # check type
        if type(value) != dict :
            raise TypeError("The spectrum CRp model should be a dictionary containing the name key and relevant parameters")

        # Continue if ok
        Spar = self._validate_spectrum_model_parameters(value, '')
        self._spectrum_crp_model = Spar

        # Information
        if not self._silent: print("Setting spectrum_crp_model value")

    @spectrum_cre1_model.setter
    def spectrum_cre1_model(self, value):
        # check type
        if type(value) != dict :
            raise TypeError("The spectrum CRe1 model should be a dictionary containing the name key and relevant parameters")

        # Continue if ok
        Spar = self._validate_spectrum_model_parameters(value, '')
        self._spectrum_cre1_model = Spar

        # Information
        if not self._silent: print("Setting spectrum_cre1_model value")

    @cre1_loss_model.setter
    def cre1_loss_model(self, value):
        # Check type
        if type(value) != str:
            raise TypeError("The CRe1 loss model should be a string")

        # Check value
        llist = ['None', 'Steady', 'Initial']
        if not value in llist:
            print('CRe1 loss model available models are:')
            print(llist)
            raise ValueError("This loss model is not available")
        
        # Setting parameters
        self._cre1_loss_model = value
        
        # Information
        if not self._silent: print("Setting cre1_loss_model value")
        
    #========== Sampling
    @Npt_per_decade_integ.setter
    def Npt_per_decade_integ(self, value):
        # Check type
        if type(value) != int:
            raise TypeError("The number of point per decade for integration should be a int")

        # Check value
        if value < 1:
            raise ValueError("The number of point per decade should be >= 1")

        # Set parameters
        self._Npt_per_decade_integ = value
        
        # Information
        if not self._silent: print("Setting number of point per decade (for integration) value")
        
    @map_coord.setter
    def map_coord(self, value):
        err_msg = ("The coordinates can be a coord object, "
                   "or a {'RA','Dec'} or {'Glon', 'Glat'} dictionary.")
        
        # Case value is a SkyCoord object
        if type(value) == astropy.coordinates.sky_coordinate.SkyCoord:
            self._map_coord = value

        # Case value is standard coordinates
        elif type(value) == dict:
            
            # It is not possible to have both RA-Dec and Glat-Glon, or just RA and not Dec, etc
            cond1 = 'RA'  in list(value.keys()) and 'Glat' in list(value.keys())
            cond2 = 'RA'  in list(value.keys()) and 'Glon' in list(value.keys())
            cond3 = 'Dec' in list(value.keys()) and 'Glat' in list(value.keys())
            cond4 = 'Dec' in list(value.keys()) and 'Glon' in list(value.keys())
            if cond1 or cond2 or cond3 or cond4:
                raise ValueError(err_msg)
            
            # Case where RA-Dec is used
            if 'RA' in list(value.keys()) and 'Dec' in list(value.keys()):
                self._map_coord = SkyCoord(value['RA'], value['Dec'], frame="icrs")

            # Case where Glon-Glat is used
            elif 'Glon' in list(value.keys()) and 'Glat' in list(value.keys()):
                self._map_coord = SkyCoord(value['Glon'], value['Glat'], frame="galactic")

            # Otherwise, not appropriate value
            else:
                raise TypeError(err_msg)

        # Case value is not accepted
        else:
            raise TypeError(err_msg)

        # Header to None
        self._map_header = None

        # Information
        if not self._silent: print("Setting the map coordinates")
        if not self._silent: print("Setting: map_header to None, as map properties are now set by hand")

    @map_reso.setter
    def map_reso(self, value):
        # check type
        try:
            test = value.to('deg')
        except:
            raise TypeError("The map resolution should be a quantity homogeneous to deg.")

        # check value
        if type(value.value) != float and type(value.value) != int and type(value.value) != np.float64:        
            raise TypeError("The map resolution should be a scalar, e.i. reso_x = reso_y.")

        # Set parameters
        self._map_reso = value
        self._map_header = None
        
        # Information
        if not self._silent: print("Setting the map resolution value")
        if not self._silent: print("Setting: map_header to None, as map properties are now set by hand")

    @map_fov.setter
    def map_fov(self, value):
        # check type
        try:
            test = value.to('deg')
        except:
            raise TypeError("The map field of view should be a quantity homogeneous to deg.")

        # Set parameters for single value application
        if type(value.value) == float or type(value.value) == int or type(value.value) == np.float64 :        
            self._map_fov = [value.to_value('deg'), value.to_value('deg')] * u.deg
    
        # Set parameters for single value application
        elif type(value.value) == np.ndarray:
            # check the dimension
            if len(value) == 2:
                self._map_fov = value
            else:
                raise TypeError("The map field of view is either a scalar, or a 2d list quantity.")

        # No other options
        else:
            raise TypeError("The map field of view is either a scalar, or a 2d list quantity.")

        # Set extra parameters
        self._map_header = None

        # Information
        if not self._silent: print("Setting the map field of view")
        if not self._silent: print("Setting: map_header to None, as map properties are now set by hand")

    @map_header.setter
    def map_header(self, value):
        # Check the header by reading it with WCS
        try:
            w = WCS(value)
            data_tpl = np.zeros((value['NAXIS2'], value['NAXIS1']))            
            header = w.to_header()
            hdu = fits.PrimaryHDU(header=header, data=data_tpl)
            header = hdu.header
        except:
            raise TypeError("It seems that the header you provided is not really a header, or does not contain NAXIS1,2.")
        
        # set the value
        self._map_header = header
        self._map_coord  = None
        self._map_reso   = None
        self._map_fov    = None

        # Information
        if not self._silent: print("Setting the map header")
        if not self._silent: print("Setting: map_coord, map_reso, map_fov to None, as the header will be used")
