import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/kman')
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *
from KMAN.Regressors import *


# Set hyper parameter(s).
N = int( Abound/2 )     # Number of anchors.
M = 3*N                 # Number of vehicles.
Nr = 3                  # Number of anchor sets + reflection sets.


# Anchor set.
# A = np.array( [[2,4,6],[3,5,7]] )
A = Abound*np.random.rand( 2,N )
# A = np.array( [
#     [i for i in range( int( Abound + 1 ) ) if i%2 != 0],
#     [i for i in range( int( Abound + 1 ) ) if i%2 != 0]] )
# A = noise( eps=Abound, shape=(2,N) )
print( 'A:\n', A )


# Reflection sets.
Ax = Rx@A
Ay = Ry@A
Q = np.hstack( (A, Ax, Ay) ).T.reshape( M,2,1 )
print( 'Ax:\n', Ax )
print( 'Ay:\n', Ay )
print( 'Q:\n', Q[:,:,0].T )


# Conversion matrix.
b = -1/4*np.array( [
    [ 1/np.sum( [A[0,j] for j in range( N ) if i != j] ) for i in range( N ) ],
    [ 1/np.sum( [A[1,j] for j in range( N ) if i != j] ) for i in range( N ) ]
] )
B = np.hstack( (b, b, b) )
print( 'B:\n', B )

z = np.array( [
    np.hstack( ([1 for i in range( N )], [0 for i in range( N )], [-1 for i in range( N )]) ),
    np.hstack( ([1 for i in range( N )], [-1 for i in range( N )], [0 for i in range( N )]) )
] )
Z = np.array( [z*b[:,None] for b in B.T] )
print( 'Z:\n', Z )


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

    # Swarm variables.
    R = -0.35
    fig, axs = plt.subplots()
    swrm = Swarm2D( X, fig=fig, axs=axs, zorder=100,
        radius=R, color='cornflowerblue' ).draw()
    anchors = Swarm2D( Q[:,:,0].T, fig=fig, axs=axs, zorder=10,
        radius=offset, draw_tail=False, color='none'
    ).setLineStyle( ':', body=True ).setLineWidth( 1.0, body=True ).draw()
    axs.plot( Q[0], Q[1], zorder=50, color='k', linestyle='none', marker='x' )

    # Axis setup.
    axs.axis( (offset+1)*np.array( [-1, 1, -1, 1] ) )
    # axs[0].set_xticks( [-i for i in range( -offset,offset+1 ) if (i % 2) != 0] )
    # axs[0].set_yticks( [-i for i in range( -offset,offset+1 ) if (i % 2) != 0] )
    axs.axis( 'equal' )
    plt.show( block=0 )

    # Environment block.
    T = 10;  Nt = round( T/dt ) + 1
    for t in range( Nt ):
        # Take measurements.
        h = anchorMeasure( X, X, eps=eps )**2
        for i in range( 1,N+1 ):
            h[(i-1)*Nr:i*Nr,(i-1)*Nr:i*Nr] = np.zeros( (Nr,Nr) )
        H = h.T.reshape( M,M,1 )

        # Apply dynamics.
        U = (Q - Z@H)[:,:,0].T
        X = model( X, C@U )

        # Update simulation.
        swrm.update( X )
        plt.pause( 1e-6 )

    # Calculate transformation matrix by DMD.
    Qerr = np.vstack( (Q, np.ones( (1,M) )) )
    regr = Regressor( Qerr, X )
    T, _ = regr.dmd()

    # Calculate error after transformation.
    print( '\nError: ', np.linalg.norm( X - T@Qerr ) )
    input( "Press ENTER to exit program... " )
