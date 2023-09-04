import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *


# Anchor values.
N = 2    # Number of anchors.
M = 3*N  # Number of vehicles.

# Anchor and corresponding reflection sets.
A = Abound*np.random.rand( 2,N )
Ax = Rx@A
Ay = Ry@A

print( 'A:\n', A )
print( 'Ax:\n', Ax )
print( 'Ay:\n', Ay )

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
    # Initialize vehicle positions.
    X = np.hstack( (
        A + noise( eps=0.1, shape=(2,N) ),
        Ax + noise( eps=0.1, shape=(2,N) ),
        Ay + noise( eps=0.1, shape=(2,N) ),
    ) )
