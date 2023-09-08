import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/kman')
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *
from KMAN.Regressors import *


# Set hyper parameter(s).
Nr = 3                      # Number of anchor sets + reflection sets.
N = np.random.randint(1,8)  # Number of anchors.
# N = 3
M = Nr*N                    # Number of vehicles.


# Exclusion elements in measurement function.
def exclude(i, j):
    return (j - i) % N == 0


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
Q = np.hstack( (A, Ax, Ay) )
print( 'Ax:\n', Ax )
print( 'Ay:\n', Ay )
print( 'Q:\n', Q )


# Conversion matrices.
Z = np.array( [
    np.hstack( ([1 for i in range( N )], [0 for i in range( N )], [-1 for i in range( N )]) ),
    np.hstack( ([1 for i in range( N )], [-1 for i in range( N )], [0 for i in range( N )]) ) ] )
print( 'Z:\n', Z )

b = -1/4*np.array( [
    [ 1/np.sum( [A[0,j] for j in range( N ) if i != j] ) for i in range( N ) ],
    [ 1/np.sum( [A[1,j] for j in range( N ) if i != j] ) for i in range( N ) ] ] )
print( 'b:\n', b )
B = np.hstack( (b, b, b) )
print( 'B:\n', B )


# Main execution block.
if __name__ == '__main__':
    # Initialize vehicle positions.
    delta = 4.0
    eps = 0.0
    X = np.hstack( (
        A + noiseCirc( eps=delta, N=N ),
        Ax + noiseCirc( eps=delta, N=N ),
        Ay + noiseCirc( eps=delta, N=N ) ) )
    print( 'Xi:\n', X )

    # Swarm variables.
    R = -0.40
    fig, axs = plt.subplots()
    swrm = Swarm2D( X, fig=fig, axs=axs, zorder=100,
        radius=R, color='cornflowerblue' ).draw()
    anchors = Swarm2D( Q, fig=fig, axs=axs, zorder=10,
        radius=delta, draw_tail=False, color='none'
        ).setLineStyle( ':', body=True
        ).setLineWidth( 1.0, body=True ).draw()
    axs.plot( Q[0], Q[1], zorder=50, color='k', linestyle='none', marker='x' )

    # Axis setup.
    axs.axis( 'equal' )
    axs.axis( 2*Abound*np.array( [-1, 1, -1, 1] ) )
    # axs[0].set_xticks( [-i for i in range( -offset,offset+1 ) if (i % 2) != 0] )
    # axs[0].set_yticks( [-i for i in range( -offset,offset+1 ) if (i % 2) != 0] )
    plt.show( block=0 )

    # Environment block.
    T = 10;  Nt = round( T/dt ) + 1
    for t in range( Nt ):
        # Take measurements.
        H = anchorMeasure( X, X, eps=eps, exclude=exclude )**2
        h = Z@H

        # Apply dynamics.
        U = C@(Q - B*h)
        X = model( X, U )

        # Impulse control.
        if t > 250 and t < 500:
            W = 0
            P = 2
            X[:,:P] = model( X[:,:P].reshape(Nx,P), W*np.ones( (Nx,P) ) )

        # Update simulation.
        swrm.update( X )
        axs.set_title( 'time: %s' % t )
        plt.pause( 1e-6 )

    # Calculate transformation matrix by DMD.
    Xerr = np.vstack( (X, np.ones( (1,M) )) )  # For consistency.
    Qerr = np.vstack( (Q, np.ones( (1,M) )) )
    regr = Regressor( Qerr, Xerr )
    T, _ = regr.dmd()
    print( 'T:\n', T )

    # Calculate error after transformation.
    print( '\nError: ', np.linalg.norm( Xerr - T@Qerr ) )
    input( "Press ENTER to exit program..." )
