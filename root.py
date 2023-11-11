from args import *

# True system values.
Abound = 10
Nx = 2
Nu = 2
dt = 0.01

# Calculate simulation step frequency.
if dt < dtsim:
    n = round( dtsim/dt )
else:
    n = 1

# Controller gains.
W = 5.0
C = W*np.eye( Nx )

# 2-D rotation by theta.
def rotz(theta):
    R = np.array( [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
        ] )
    return R

# Model function.
def model(X, U):
    return X + dt*U

# Calculate anchor coefficient matrices.
def distanceBasedControlMatrices( Aset, N ):
    A, B = anchorDifferenceMatrices(Aset, N=N)
    Z, _ = Regressor( A.T@A, np.eye( Nx,Nx ) ).dmd()
    K = Z@A.T
    return C, K, B

# Shaped noise functions.
# Boxed noise.
def noise(eps=1e-3, shape=(2,1), shape3=None):
    if shape3 is not None:
        return 2*eps*np.random.rand( shape[0], shape[1], shape[2] ) - eps
    return 2*eps*np.random.rand( shape[0], shape[1] ) - eps

# Noise in radius.
def circ(R, t):
    return np.array( [R*np.cos( t ), R*np.sin( t )] ).reshape( 2,1 )

def noiseCirc(eps=1e-3, N=1):
    y = np.empty( (2,N) )
    for i in range( N ):
        t = 2*np.pi*np.random.rand()
        R = eps*np.random.rand()
        y[:,i] = circ( R, t )[:,0]
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
        d = d + noise( eps=eps, shape=(N,M) )
        d[d < 0] = 0  # Distance cannot be negative from noise.
    return np.sqrt( d )

def anchorMeasureStack(D):
    N = D.shape[1]
    h = np.empty( (N*N,1) )
    pq = 0
    for dp in D.T:
        for dq in D.T:
            h[pq] = dp**2 - dq**2
            pq += 1
    return h

def vehicleMeasureStack(X, A, eps=0):
    N = A.shape[1]
    M = X.shape[1];
    H = np.zeros( (N*N,M) )
    for i, x in enumerate( X.T ):
        pq = 0
        D = anchorMeasure( x[:,None], A, eps=eps )
        H[:,i] = anchorMeasureStack( D )[:,0]
    return H

# Control-related members.
def centroid(X):
    n = X.shape[1]
    Xbar = 1/n*np.sum( X, axis=1 )
    return Xbar[:,None]

def rotation(X, Y):
    C = X@Y.T
    U, _, V = np.linalg.svd( C )
    d = np.sign( np.linalg.det( V@U.T ) )
    R = V@[[1, 0],[0,d]]@U.T
    return R

def lyapunovCandidate( X, A ):
    n = X.shape[1]

    Xbar = centroid( X )
    Abar = centroid( A )
    Psi = rotation( X - Xbar, A - Abar )

    V = 0
    for x, a in zip( X.T, A.T ):
        xerr = Psi@( x[:,None] - Xbar ) - (a[:,None] - Abar)
        V += xerr.T@xerr

    print( V )
    return V

def lyapunovCandidateAnchored( X, A, R=np.eye( Nx,Nx ), r=0 ):
    V = 0
    for x, a in zip( X.T, A.T ):
        V += (x[:,None] - (R@a[:,None] + r)).T@(x[:,None] - (R@a[:,None] + r))
    return V

# Per-vehicle candidate function.
def lyapunovCandidatePerVehicle(N, t, X, Xeq):
    Vsnap = np.hstack( [
        lyapunovCandidateAnchored( x[:,None], Xeq )
        for x in X.T] )
    V = np.vstack( (t*np.ones( (1,N) ), Vsnap) )
    return V

def anchorDifferenceMatrices(Aset, N=1):
    # Non-squared coefficient matrix.
    A = -2*np.vstack( ( [
        [ap - aq for aq in Aset.T]
            for ap in Aset.T ] ) )

    # Squared-norm matrix.
    b = np.vstack( [
        [ap@ap[:,None] - aq@aq[:,None] for aq in Aset.T]
            for ap in Aset.T ] )
    B = np.kron( b, np.ones( (1,N) ))

    # Return matrices.
    return A, B

def distanceBasedControl(X, Xeq, C, K, B, A=None, eps=0):
    if A is None:
        A = X
    H = vehicleMeasureStack( X, A, eps=eps )
    Y = K@(H - B)
    return -C@(Y - Xeq), Y
