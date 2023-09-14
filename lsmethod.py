# Import library functions.
from root import *


# Dimensional variables.
n = Nx
N = 3
P = N*N

# Anchor set initialization.
A = np.array( [[-1, 1, 1],[1, 1, -1]] )
print( 'A:\n', A )

# Anchor-based control coefficients.
B = -2*np.vstack( (
    [ [ap - aq
        for p, ap in enumerate( A.T ) if p!=q]
            for q, aq in enumerate( A.T ) ]
) )
Z, _ = Regressor( B.T@B, np.eye( n,n ) ).dmd()
Ka = Z@B.T
print( 'Ka:\n', Ka )

# Measurement stack for control synthesis.
def anchorMeasureStack( X, A ):
    z = anchorMeasure( X, A )
    H = np.vstack( (
        [ [dp**2 - dq**2 - ap@ap[:,None] + aq@ap[:,None]
            for p, (dp, ap) in enumerate( zip( z.T, A.T ) ) if p!=q]
                for q, (dq, aq) in enumerate( zip( z.T, A.T ) ) ]
    ) )
    return H

# Testing.
X = noise( eps=Abound, shape=(Nx,N) )
H = anchorMeasureStack( X, A )
print( 'H:\n', H )
print( 'X is\n', X )
print( 'X is approximately\n', Ka@H )
