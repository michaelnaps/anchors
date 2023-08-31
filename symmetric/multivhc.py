import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *


# Controller gains.
C = np.eye( Nx )
# C = 10*np.diag( np.random.rand( Nx, ) )

# Symmetric tranformations.
Rx = np.array( [ [-1, 0], [0, 1] ] )
Ry = np.array( [ [1, 0], [0, -1] ] )

# Anchor values.
# Na = 1
# Na = 100
Na = np.random.randint( 1,15 )

# Desired position term.
q = 7.5*np.ones( (Nx,1) )
# q = 2*A*np.random.rand( 2,1 ) - A
print( 'number of anchors: ', 3*Na )
print( 'desired position: ', q.T )

# Anchor and reflection sets.
aList = noise( eps=A, shape=(2,Na) )
rxList = Rx@aList
ryList = Ry@aList

# Control matrices.
D = -1/4*np.diag( [ 1/np.sum( aList[0] ), 1/np.sum( aList[1] ) ] )
print( 'Anchor coefficient matrix:\n', D )
