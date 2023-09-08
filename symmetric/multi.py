import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/kman')
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *
from KMAN.Regressors import *


# Anchor values.
# N = int( Abound/2 )    # Number of anchors.
N = 3
M = 3*N                # Number of vehicles.

# Anchor and corresponding reflection sets.
# A = Abound*np.random.rand( 2,N )
A = np.array( [
    [2, 4, 6],
    [3, 5, 7]
] )
# A = np.array( [
#     [i for i in range( int( Abound + 1 ) ) if i%2 != 0],
#     [i for i in range( int( Abound + 1 ) ) if i%2 != 0]] )
# A = noise( eps=Abound, shape=(2,N) )
Ax = Rx@A
Ay = Ry@A
ANCHORS = np.hstack( (A, Ax, Ay) )
print( 'A:\n', A )
print( 'Ax:\n', Ax )
print( 'Ay:\n', Ay )
print( 'Q:', ANCHORS )


# Anchor measurement functions.
def anchorMeasure(x, aList=None, eps=None, exclude=[-1]):
    if aList is None:
        aList = A
    Na = aList.shape[1]
    d = np.zeros( (1,Na) )
    for i, a in enumerate( aList.T ):
        if i not in exclude:
            d[:,i] = (x - a[:,None]).T@(x - a[:,None])
    if eps is not None:
        d = d + np.abs( noise( eps=eps, shape=(1,Na) ) )
    return np.sqrt( d )


# Main execution block.
if __name__ == '__main__':
    # Initialize vehicle positions.
    eps = None
    offset = 0.0
    X = np.hstack( (
        A + noiseCirc( eps=offset, N=N ),
        Ax + noiseCirc( eps=offset, N=N ),
        Ay + noiseCirc( eps=offset, N=N ),
    ) )
    # X = np.hstack( (
    #     A + offset*np.ones( (2,1) ),
    #     Ax + Rx@(offset*np.ones( (2,1) )),
    #     Ay + Ry@(offset*np.ones( (2,1) )),
    # ) )
    print( 'Xi:\n', X )

    # Initialize control sets.
    k = 0
    zSet = np.empty( (M,2,2) )
    qSet = np.empty( (M,2,1) )
    for i, x in enumerate( X.T ):
        if k == N:  k = 0
        Atemp = np.array( [A[:,j] for j in range( N ) if k != j] ).T
        zSet[i] = -1/4*np.diag( [ 1/np.sum( Atemp[0] ), 1/np.sum( Atemp[1] ) ] )
        qSet[i] = ANCHORS[:,i,None]
        k += 1
    # print( 'Z:\n', zSet )
    # print( 'Q:\n', qSet )

    print( 'zSet:\n', zSet[:,:,0].T )
    print( np.vstack( (zSet[:,:,0].T[0], zSet[:,:,1].T[1]) ) )

    # Swarm variables.
    R = -0.35
    fig, axs = plt.subplots()
    swrm = Swarm2D( X, fig=fig, axs=axs,
        radius=R, color='cornflowerblue' ).draw()

    # Draw anchors for reference.
    axs.plot( ANCHORS[0], ANCHORS[1], color='k', linestyle='none', marker='x' )
    anchors = Swarm2D( ANCHORS, fig=fig, axs=axs,
        radius=offset, draw_tail=False, color='none'
    ).setLineStyle( ':', body=True ).setLineWidth( 1.0, body=True ).draw()

    # Axis setup.
    axs.axis( (offset+1)*np.array( [-1, 1, -1, 1] ) )
    # axs[0].set_xticks( [-i for i in range( -offset,offset+1 ) if (i % 2) != 0] )
    # axs[0].set_yticks( [-i for i in range( -offset,offset+1 ) if (i % 2) != 0] )
    axs.axis( 'equal' )
    plt.show( block=0 )

    TEMPMAT = np.empty( (M,M) )
    TEMPLIST = np.empty( (2,M) )

    print( '---' )
    j = 0
    U = np.empty( (Nu,M) )
    for i, (x, q, Z) in enumerate( zip( X.T, qSet, zSet ) ):
        if j == N:  j = 0
        D = anchorMeasure( x[:,None], aList=X[:,:N], eps=eps, exclude=[j] )
        Dx = anchorMeasure( x[:,None], aList=X[:,N:2*N], eps=eps, exclude=[j] )
        Dy = anchorMeasure( x[:,None], aList=X[:,2*N:3*N], eps=eps, exclude=[j] )
        z = np.vstack( (
            np.sum( D**2 - Dy**2, axis=1 ),
            np.sum( D**2 - Dx**2, axis=1 )
        ) )

        TEMPMAT[:,i] = np.vstack( (D.T, Dx.T, Dy.T) )[:,0]
        TEMPLIST[:,i] = z[:,0]

        U[:,i] = (C@(q - Z@z))[:,0]
        X[:,i] = model( x[:,None], U[:,i,None] )[:,0]
        j += 1

    print( 'H:\n', TEMPMAT**2 )
    print( 'h:\n', TEMPLIST )
    print( U )
    print( X )

    exit()

    # Simulation block.
    T = 10;  Nt = round( T/dt ) + 1
    for t in range( Nt ):
        j = 0
        for i, (x, q, Z) in enumerate( zip( X.T, qSet, zSet ) ):
            if j == N:  j = 0
            D = anchorMeasure( x[:,None], aList=X[:,:N], eps=eps, exclude=[j] )
            Dx = anchorMeasure( x[:,None], aList=X[:,N:2*N], eps=eps, exclude=[j] )
            Dy = anchorMeasure( x[:,None], aList=X[:,2*N:3*N], eps=eps, exclude=[j] )
            z = np.vstack( (
                np.sum( D**2 - Dy**2, axis=1 ),
                np.sum( D**2 - Dx**2, axis=1 )
            ) )
            U = C@(q - Z@z)
            X[:,i] = model( x[:,None], U )[:,0]
            j += 1

        if t > 250 and t < 300:
            P = 0
            w = 5.0
            X[:,:P] = model( X[:,:P], w*np.ones( (Nx,P) ) )

        swrm.update( X )
        axs.set_title( 't = %s' % t )
        plt.pause( 1e-6 )

        # if np.linalg.norm( U ) < 1e-6:
        #     print( 'Stopping simulation. No movement...' )
        #     break

    # Calculate transformation matrix by DMD.
    Aerr = np.vstack( (ANCHORS, np.ones( (1,M) )) )
    regr = Regressor( Aerr, X )
    T, _ = regr.dmd()

    # Calculate error after transformation.
    print( '\nError: ', np.linalg.norm( X - T@Aerr ) )
    input( "Press ENTER to exit program... " )
