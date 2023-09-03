import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *


# Anchor values.
N = 2    # Number of anchors.
M = 3*N  # Number of vehicles.

# Desired position set.
Q = np.array(
    [
        [-1, 1, 1],
        [1, 1, -1]
    ],
    [
        [-2, 2, 2],
        [2, 2, -2]
    ] )

# Anchor and reflection sets.
aList = noise( eps=A, shape=(2,Na) )
axList = Rx@aList
ayList = Ry@aList

# Control matrices.
D = -1/4*np.diag( [ 1/np.sum( aList[0] ), 1/np.sum( aList[1] ) ] )
print( 'Anchor coefficient matrix:\n', D )


# Main execution block.
if __name__ == '__main__':
    pass
