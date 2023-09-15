# Import library functions.
from root import *


# Dimensional variables.
n = Nx
N = 3
P = N*N

# Exclusion function.
exclude = lambda i, j: i == j

# Anchor set initialization.
# A = np.array( [[-1, 1, 1],[1, 1, -1]] )
# A = np.array( [[-1, 2, 3],[4, 5, -6]] )
A = noise( eps=Abound, shape=(n,N) )
print( 'A:\n', A )

# Anchor-based control coefficients.
B = 2*np.vstack( (
    [ [ap - aq
        for p, ap in enumerate( A.T ) if p != q]
            for q, aq in enumerate( A.T ) ]
) )
Z, _ = Regressor( B.T@B, np.eye( n,n ) ).dmd()
Ka = Z@B.T
print( 'Ka:\n', Ka )

# Measurement stack for control synthesis.
def anchorMeasureStack(x, A, D):
    N = A.shape[1];  pq = 0
    h = np.empty( (N*(N-1),1) )
    for p, (dp, ap) in enumerate( zip( D.T, A.T ) ):
        for q, (dq, aq) in enumerate( zip( D.T, A.T ) ):
            if p != q:
                h[pq] = dp**2 - dq**2 - ap@ap[:,None] + aq@aq[:,None]
                pq += 1
    return h

def vehicleMeasureStack(X, A):
    M = X.shape[1];  N = A.shape[1]
    H = np.zeros( (N*(N-1),M) )
    for i, x in enumerate( X.T ):
        pq = 0
        D = anchorMeasure( x[:,None], A )
        H[:,i] = anchorMeasureStack( x, A, D ).T
    return H

# Testing.
X = noise( eps=Abound, shape=(Nx,N) )
H = vehicleMeasureStack( X, A )
print( 'X is\n', X )
print( 'X is approximately\n', Ka@H )
