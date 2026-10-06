from . import nn_utils as _nn_utils
from .nn_utils import *
from .pcd_manipulation import *
from .knn_torch import *


def __getattr__(name):
    return getattr(_nn_utils, name)
