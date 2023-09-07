import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/kman')
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *
from KMAN.Regressors import *


# Set hyper parameter(s).
# N = int( Abound/2 )    # Number of anchors.
N = 2
M = 3*N                 # Number of vehicles.
Nr = 3                  # Number of anchor sets + reflection sets.


# Anchor set.
A = Abound*np.random.rand( 2,N )
# A = np.array( [
#     [i for i in range( int( Abound + 1 ) ) if i%2 != 0],
#     [i for i in range( int( Abound + 1 ) ) if i%2 != 0]] )
# A = noise( eps=Abound, shape=(2,N) )
print( 'A:\n', A )


# Reflection sets.
Ax = Rx@A
Ay = Ry@A
print( 'Ax:\n', Ax )
print( 'Ay:\n', Ay )


# Conversion matrix.
Asum = np.sum( A, axis=1 )
Z = -1/(4*Asum[:,None])*np.array( [
    np.hstack( ([1 for i in range( N )], [0 for i in range( N )], [-1 for i in range( N )]) ),
    np.hstack( ([1 for i in range( N )], [-1 for i in range( N )], [0 for i in range( N )]) )
] )
Q = np.hstack( (A, Ax, Ay) )

print( 'Z:\n', Z )


# Main execution block.
if __name__ == '__main__':
    # Initialize vehicle positions.
    eps = None
    offset = 2.5
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
    print( 'Xi:\n', X.T )

    # Swarm variables.
    R = -0.35
    fig, axs = plt.subplots()
    swrm = Swarm2D( X, fig=fig, axs=axs, zorder=100,
        radius=R, color='cornflowerblue' ).draw()
    anchors = Swarm2D( Q, fig=fig, axs=axs, zorder=10,
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
        h = anchorMeasure( X, X, eps=eps )
        for i in range( 1,Nr ):
            h[(i-1)*Nr:i*Nr,(i-1)*Nr:i*Nr] = np.zeros( (Nr,Nr) )

        # Apply dynamics.
        print( h )
        U = C@(Q - Z@h)
        X = model( X, U )

        # Update simulation.
        swrm.update( X )
        plt.pause( 1e-6 )

    # Calculate transformation matrix by DMD.
    Aerr = np.vstack( (ANCHORS, np.ones( (1,M) )) )
    regr = Regressor( Aerr, X )
    T, _ = regr.dmd()

    # Calculate error after transformation.
    print( '\nError: ', np.linalg.norm( X - T@Aerr ) )
    input( "Press ENTER to exit program... " )
