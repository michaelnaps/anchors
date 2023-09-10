import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/geom')
sys.path.insert(0, expanduser('~')+'/prog/kman')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument( '--sim' )
parser.add_argument( '--dtsim' )
parser.add_argument( '--pausesim' )
args = parser.parse_args()

sim = args.sim
dtsim = args.dtsim
pausesim = args.pausesim

if dtsim is None:
    dtsim = 1e-6

if pausesim is None:
    pausesim = 1e-12