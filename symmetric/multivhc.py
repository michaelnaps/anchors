import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *


# Anchor values.
N = 2    # Number of anchors.
M = 3*N  # Number of vehicles.

# Desired position set (in A).
Q = np.array( [
    [ [1], [1] ],
    [ [2], [2] ] ] )

# Anchor and corresponding reflection sets.
A = [ q + noise( eps=1e-1, shape=(2,1) ) for q in Q[::-1] ]
Ax = [ Rx@a for a in A ]
Ay = [ Ry@a for a in A ]

# Anchor-distance control matrices.
D = np.array( [ -1/4*np.diag( [ 1/np.sum( a[0] ), 1/np.sum( a[1] ) ] ) for a in A ] )
print( 'Anchor coefficient matrices:\n', D )


# Anchor measurement functions.
def anchorMeasure(x):
    d = np.empty( (1,Na) )
    for i, a in enumerate( A.T ):
        d[:,i] = (x - a[:,None]).T@(x - a[:,None])
    return np.sqrt( d )

def reflectionMeasure(x):
    dr = np.empty( (2,Na) )
    for i, (ax, ay) in enumerate( zip( Ax.T, Ay.T ) ):
        dr[:,i] = np.array( [
            (x - ax[:,None]).T@(x - ax[:,None]),
            (x - ay[:,None]).T@(x - ay[:,None]),
        ] )[:,0,0]
    return np.sqrt( dr )


# Main execution block.
if __name__ == '__main__':
    pass
