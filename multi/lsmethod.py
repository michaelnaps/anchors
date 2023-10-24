# Add path.
import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

# Import related functions.
from root import *


# Dimensional variables.
N = 3
P = N*N

# Exclusion function.
exclude = lambda i, j: i == j

# Anchor set initialization.
# A = np.array( [[-1, 1, 1],[1, 1, -1]] )
# A = np.array( [[-1, 2, 3],[4, 5, -6]] )
Xeq = noise( eps=Abound, shape=(Nx,N) )
print( 'Xeq:\n', Xeq )

# Anchor position-related matrices.
A, B = anchorDifferenceMatrices( Xeq, N=N )
print( 'A:', A )
print( 'B:', B )

# Anchor-based control coefficients.
Z, _ = Regressor( A.T@A, np.eye( Nx,Nx ) ).dmd()
K = Z@A.T
print( 'K:\n', K )

# Testing.
X = noise( eps=Abound, shape=(Nx,N) )
H = vehicleMeasureStack( X, Xeq )
print( 'X is\n', X )
print( 'X is approximately\n', K@(H - B) )
