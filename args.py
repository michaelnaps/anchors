# System imports.
import argparse
import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/geom')
sys.path.insert(0, expanduser('~')+'/prog/kman')

# Environment imports.
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from GEOM.Vehicle2D import *
from KMAN.Regressors import *


# Figure filepath.
figurepath = expanduser('~') \
    + '/Documents/anchors_paper/figures/'

# Plot font.
default_cycler = cycler(
    color=[
        'cornflowerblue',
        'indianred',
        'mediumpurple',
        'sandybrown',
        'yellowgreen',
        'steelblue'
    ] )
plt.rcParams.update( {
    'axes.prop_cycle': default_cycler,
    'text.usetex': True,
    'font.family': 'mathptmx',
    'font.size': 14,
    'text.latex.preamble': "\\usepackage{amsmath}",
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
save = args.save == '1'
sim = args.sim == '1' and not save

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
