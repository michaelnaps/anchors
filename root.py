from args import *
from matplotlib.lines import Line2D


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

# Symmetric tranformations.
Rx = np.array( [ [1, 0], [0, -1] ] )
Ry = np.array( [ [-1, 0], [0, 1] ] )

def PSI(A):
    n = A.shape[1]
    return np.vstack( (A, np.ones( (1,n) )) )


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

# Noise in radius.
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
    return 1/n*np.sum( X, axis=1 )[:,None]

def formationError( X, Xeq ):
    Q = np.vstack(
        (Xeq, np.ones( (1,Xeq.shape[1]) )) )
    regr = Regressor( Q, X )
    T, _ = regr.dmd()
    return T, regr.err

def lyapunovCandidate( X, A ):
    n = X.shape[1]

    Xbar = centroid( X )
    Abar = centroid( A )
    Psi, _ = Regressor( X - Xbar, A - Abar ).dmd()

    V = 0
    for x, a in zip( X.T, A.T ):
        xerr = Psi@( x[:,None] - Xbar ) - (a[:,None] - Abar)
        V += (xerr.T@xerr)[0][0]

    return V

def signedCoefficientMatrix(N):
    # Signed coefficient matrix.
    S = np.array( [
        np.hstack( ([1 for i in range( N )], [0 for i in range( N )], [-1 for i in range( N )]) ),
        np.hstack( ([1 for i in range( N )], [-1 for i in range( N )], [0 for i in range( N )]) ) ] )
    return S

def anchorCoefficientMatrix(A, N, exclude=None):
    if exclude is None:
        exclude = lambda i, j: False
    # Anchor-position coefficients.
    z = -1/4*np.hstack( [
        [ 1/np.sum( np.hstack( [A[:,j,None]
            for j in range( N ) if not exclude(i,j)] ), axis=1 )
                for i in range( N ) ] ] ).T
    Z = np.hstack( (z, z, z) )
    return Z

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

def symmetricControl(X, Q, C, Z, S, K=None, eps=0, exclude=lambda i,j: False):
    # Include all elements if None.
    if K is None:
        K = X.shape[1]
    # Take measurements and return control.
    H = anchorMeasure( X, X, eps=eps, exclude=exclude )**2
    return C@(Q - (Z*S)@H[:K])

def asymmetricControl(X, Q, C, K, B, eps=0):
    H = vehicleMeasureStack( X, X, eps=eps )
    return -C@(K@(H - B) - Q)


# Plotting-related members.
def initAnchorEnvironment(X, Q, A, e0, Nt=1000, ge=1, R1=0.40, R2=1.00, anchs=True, dist=True):
    # Plot initialization.
    Np = 2
    fig, axs = plt.subplots(1,Np)

    # Optionally plot anchors.
    if anchs:
        anchors = Swarm2D( Q, fig=fig, axs=axs[0], zorder=50,
            radius=R1, draw_tail=False, color='indianred'
            ).setLineStyle( None, body=True ).draw()
    else:
        anchors = None

    if dist:
        disturb = Swarm2D( Q, fig=fig, axs=axs[0], zorder=10,
            radius=R2, draw_tail=False, color='none'
            ).setLineStyle( ':', body=True
            ).setLineWidth( 1.0, body=True ).draw()
    else:
        disturb = None

    # Swarm variables.
    swrm = Swarm2D( X, fig=fig, axs=axs[0], zorder=100,
        radius=-R1, color='cornflowerblue', tail_length=Nt,
        draw_tail=sim ).draw()
    axs[0].plot( Q[0], Q[1], zorder=50, color='indianred',
        linestyle='none', marker='x' )
    axs[0].plot( X[0], X[1], zorder=50, color='cornflowerblue',
        linestyle='none', marker='x' )

    # For plotting error.
    error = Vehicle2D( e0, fig=fig, axs=axs[1],
        radius=0.0, color='cornflowerblue', tail_length=Nt ).draw()
    axs[1].plot( [0, ge*Nt], [0, 0], color='indianred', linestyle='--' )

    # Axis setup.
    titles = ('Environment', 'Lyapunov Trend')
    xlabels = ('$\\mathbf{x}$', 'Iteration')
    # ylabels = ('$\\mathbf{y}$', '$|| X - (\\Psi X^{(\\text{eq})} + \\psi) ||_2$')
    ylabels = ('$\\mathbf{y}$', '$V(\\Psi X + \\psi)$')
    bounds = np.vstack( [
        1.5*Abound*np.array( [-1, 1, -1, 1] ),
        np.hstack( [e0[0], ge*Nt, -0.01, e0[1]] ) ] )
    for i in range( 2 ):
        axs[i].set_title( titles[i] )
        axs[i].set_xlabel( xlabels[i] )
        axs[i].set_ylabel( ylabels[i] )
        axs[i].axis( bounds[i] )
    axs[0].grid( 0 )
    axs[1].grid( 1 )
    axs[0].axis( 'equal' )

    # Legend formation.
    legend_elements = [
        Line2D([0], [0], color='cornflowerblue', linestyle='none', marker='x',
            label='$X^{(0)}$'),
        Line2D([0], [0], color='cornflowerblue', marker='o', markerfacecolor='none',
            label='$X$'),
        Line2D([0], [0], color='indianred', linestyle='none', marker='x',
            label='$\\mathcal{A}$, $X^{(\\text{eq})}$' ),
        Line2D([0], [0], color='grey', linestyle='--',
            label='$\\Psi [\\mathbf{x}, \\mathbf{y}]^\\top + \\psi$' ),
    ]
    axs[1].legend( handles=legend_elements, ncol=1 )

    # Show plot.
    fig.tight_layout()
    fig.set_figheight( figheight )
    plt.show( block=0 )

    # Return figure.
    return fig, axs, swrm, anchors, error

def finalAnchorEnvironment( fig, axs, swrm, xList, eList, Psi, Xbar, shrink=1/3 ):
    if not sim:
        swrm.update( xList[:,-1,:].T )
        axs[1].plot( eList[0], eList[1], color='cornflowerblue' )
    for vhc in xList:
        axs[0].plot( vhc.T[0], vhc.T[1], color='cornflowerblue' )

    xaxis = shrink*Psi@( [[-Abound, Abound],[0, 0]] + Xbar )
    yaxis = shrink*Psi@( [[0, 0],[-Abound, Abound]] + Xbar )

    axs[0].plot( xaxis[0], xaxis[1], color='grey', linestyle='--', zorder=150 )
    axs[0].plot( yaxis[0], yaxis[1], color='grey', linestyle='--', zorder=150 )
    axs[0].plot( xaxis[0,1], xaxis[1,1], color='grey', marker='o', zorder=150 )
    axs[0].plot( yaxis[0,1], yaxis[1,1], color='grey', marker='o', zorder=150 )
    axs[1].axis( np.array( [0.0, max( eList[0] ), 0.0, max( eList[1] )] ) )

    # Return figure.
    return fig, axs


# Operator functions (currently not being used).
def rotz(theta):
    R = np.array( [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
        ] )
    return R

def vec(A):
    m, n = A.shape
    return A.reshape( m*n,1 )

def kronsum(x1, x2):
    m1, n1 = x1.shape
    m2, n2 = x2.shape
    y = np.empty( (m1*m2, n1*n2) )
    i = 0
    for r1 in x1:
        j = 0
        for c1 in r1:
            y[i:i+m2,j:j+n2] = c1 + x2
            j += n2
        i += m2
    return y
