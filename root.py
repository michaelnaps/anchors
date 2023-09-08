import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/geom')

import numpy as np
from GEOM.Vehicle2D import *


# set global output setting
np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)


# Pause time argument.
if len( sys.argv ) > 1:
    sim = sys.argv[1] == '1'
else:
    sim = False

if len( sys.argv ) > 2:
    sim_pause = float( sys.argv[2] )
else:
    sim_pause = 1e-3


# True system values.
Abound = 10
Nx = 2
Nu = 2
dt = 0.01


# Controller gains.
W = 10.0
C = W*np.eye( Nx )

# Symmetric tranformations.
Rx = np.array( [ [1, 0], [0, -1] ] )
Ry = np.array( [ [-1, 0], [0, 1] ] )


# Model function.
def model(X, U):
    return X + dt*U

# Control function.
def control(x, C=np.eye( Nx ), q=np.zeros( (Nx,1) )):
    return C@(q - x)

# Shaped noise functions.
# Boxed noise.
def noise(eps=1e-3, shape=(2,1), shape3=None):
    if shape3 is not None:
        return 2*eps*np.random.rand( shape[0], shape[1], shape[2] ) - eps
    return 2*eps*np.random.rand( shape[0], shape[1] ) - eps

# Concentric noise.
def noiseCirc(eps=1e-3, N=1):
    y = np.empty( (2,N) )
    for i in range( N ):
        t = 2*np.pi*np.random.rand()
        R = eps*np.random.rand()
        y[:,i] = [R*np.cos( t ), R*np.sin( t )]
    return y

# Anchor measurement function.
def anchorMeasure(X, A, eps=None, exclude=lambda i,j: False):
    N = X.shape[1]
    M = A.shape[1]
    d = np.zeros( (N,M) )
    for i, x in enumerate( X.T ):
        for j, a in enumerate( A.T ):
            if not exclude(i,j):
                d[i,j] = (x[:,None] - a[:,None]).T@(x[:,None] - a[:,None])
    if eps is not None:
        d = d + np.abs( noise( eps=eps, shape=(N,M) ) )
    return np.sqrt( d )
