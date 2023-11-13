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

# Get the sub root directory (specific to my machines).
if 'linux' in sys.platform:
    subroot = '/Documents'
elif 'darwin' in sys.platform:
    subroot = '/prog'

# Command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument( '--save' )
parser.add_argument( '--show' )
parser.add_argument( '--sim' )
parser.add_argument( '--dtsim' )
parser.add_argument( '--pausesim' )
parser.add_argument( '--fheight' )
parser.add_argument( '--fontsize' )

# Program variables.
args = parser.parse_args()
save = args.save == '1'
show = args.show == '1'
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

if args.fontsize is None:
    fontsize = 14
else:
    fontsize = float( args.fontsize )

# Figure filepath.
figurepath = expanduser('~') \
    + subroot + '/anchors_paper/figures/'

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
    'font.size': fontsize,
    'text.latex.preamble': "\\usepackage{amsmath}",
} )

# Set global number print setting.
np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)