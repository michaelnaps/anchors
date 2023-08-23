import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/geom')

import numpy as np
from GEOM.Vehicle2D import *

# Model function.
def model(x, u):
    return x + dt*u