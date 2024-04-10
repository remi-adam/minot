__version__ = '1.1.3'

try:
    from .model import Cluster
except ImportError:
    print('WARNING: Could not import Cluster from model. You')
    print('         may try (re)installing dependencies by  ')
    print('         hand. For example running:              ')
    print('             $ conda install matplotlib          ')
    print('             $ conda install numpy               ')
    print('             $ conda install scipy               ')
    print('             $ conda install astropy             ')
