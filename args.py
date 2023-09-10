# System imports.
import argparse
import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/geom')
sys.path.insert(0, expanduser('~')+'/prog/kman')

# Environment imports.
import numpy as np
from GEOM.Vehicle2D import *
from KMAN.Regressors import *


# Figure filepath.
figurepath = expanduser('~') + '/prog/anchors/.figures'

# Set global number print setting.
np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)

# Command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument( '--sim' )
parser.add_argument( '--save' )
parser.add_argument( '--dtsim' )
parser.add_argument( '--pausesim' )
args = parser.parse_args()

# Program variables.
sim = args.sim == True
save = args.save == True
dtsim = args.dtsim
pausesim = args.pausesim

if dtsim is None:
    dtsim = 1e-6

if pausesim is None:
    pausesim = 1e-12
