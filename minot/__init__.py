__version__ = '1.0'

try:
    from .model import Cluster
except ImportError:
    print('WARNING: Could not import Cluster from model.')
