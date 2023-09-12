# System imports.
import argparse
import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/geom')
sys.path.insert(0, expanduser('~')+'/prog/kman')

# Environment imports.
import numpy as np
import matplotlib.pyplot as plt
from GEOM.Vehicle2D import *
from KMAN.Regressors import *


# Figure filepath.
figurepath = expanduser('~') \
    + '/bu_research/symmetric_formation_control/figures/sim/'

# Plot font.
plt.rcParams.update( {
    'text.usetex': True,
    'font.family': 'mathptmx',
    'text.latex.preamble': "\\usepackage{amsmath}",
    'font.size': 14
} )

# Set global number print setting.
np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)

# Command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument( '--save' )
parser.add_argument( '--sim' )
parser.add_argument( '--dtsim' )
parser.add_argument( '--pausesim' )
parser.add_argument( '--fheight' )

# Program variables.
args = parser.parse_args()
save = bool( args.save )
sim = bool( args.sim ) and not save

if args.dtsim is None:
    dtsim = 1e-6
else:
    dtsim = float( args.dtsim )

if args.pausesim is None:
    pausesim = 1e-12
else:
    pausesim = float( args.pausesim )

if args.fheight is None:
    figheight = 5
else:
    figheight = float( args.fheight )
