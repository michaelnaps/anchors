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

# Anchor position-related matrices.
B = -2*np.vstack( ( [
    [ap - aq for aq in A.T]
        for ap in A.T ] ) )
B2 = np.vstack( [
    [ap@ap[:,None] - aq@aq[:,None] for aq in A.T]
        for ap in A.T ] )

# Anchor-based control coefficients.
Z, _ = Regressor( B.T@B, np.eye( n,n ) ).dmd()
K = Z@B.T
print( 'K:\n', K )

# Measurement stack for control synthesis.
def anchorMeasureStack(x, D):
    h = np.empty( (P,1) )
    pq = 0
    for dp in D.T:
        for dq in D.T:
            h[pq] = dp**2 - dq**2
            pq += 1
    return h

def vehicleMeasureStack(X, A):
    M = X.shape[1];
    H = np.zeros( (P,M) )
    for i, x in enumerate( X.T ):
        pq = 0
        D = anchorMeasure( x[:,None], A )
        H[:,i] = anchorMeasureStack( x, D )[:,0]
    return H

# Testing.
X = noise( eps=Abound, shape=(Nx,N) )
H = vehicleMeasureStack( X, A )
print( 'X is\n', X )
print( 'X is approximately\n', K@(H - B2) )
