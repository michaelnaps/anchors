import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/geom')

import numpy as np
from GEOM.Vehicle2D import *

# True system values.
A = 2
dt = 0.01
Nx = 2
Nu = 2

# Model function.
def model(x, u):
    return x + dt*u

# Control function.
def control(x, C=np.eye( Nx ), q=np.zeros( (Nx,1) )):
    return C@(q - x)