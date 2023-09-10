import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *


# Set hyper parameter(s).
Nr = 3                      # Number of anchor sets + reflection sets.
N = 3                       # Number of anchors.
# N = np.random.randint(1,10)
M = Nr*N                    # Number of vehicles.

# Exclusion elements in measurement function.
def exclude(i, j):
    return False # (j - i) % N == 0

# Anchor set.
A = np.array( [[2,5,8],[3,6,9]] )
# A = Abound*np.random.rand( 2,N )
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

# For error calculation.
Qerr = np.vstack( (Q, np.ones( (1,M) )) )

# Calculate anchor coefficient matrices.
S = signedCoefficientMatrix( N )
Z = anchorCoefficientMatrix( A, N, exclude=exclude )
print( 'S:\n', S )
print( 'Z:\n', Z )


# Main execution block.
if __name__ == '__main__':
    # Simulation time.
    T = 5;  Nt = round( T/dt ) + 1
    tList = np.array( [ [i for i in range( Nt )] ] )

    # Initialize vehicle positions.
    delta = 1.0
    X0 = Q + noiseCirc( eps=delta, N=M )
    Ne = 9
    epsList = [0] + [2**i for i in range( Ne-1 )]
    print( 'eps:\n', epsList )

    # For error trend plotting.
    eTrend = np.empty( (Ne,Nt) )

    # Simulation block.
    for i, eps in enumerate( epsList ):
        # Reset initial conditions.
        X = X0

        # Initial error calculation.
        eTrend[i,0] = formationError( X, Q )[1]
        print( 'Error:', eps )
        print( 'Initial: %.3f' % eTrend[i,0] )
        for j in range( 1,Nt ):
            # Calculate control.
            U = symmetricControl( X, Q, C, Z, S, eps=eps, exclude=exclude )

            # Apply dynamics.
            X = model( X, U )

            # Calculate tranformation error.
            eTrend[i,j] = formationError( X, Q )[1]
        print( 'Final Error: %.3e' % eTrend[i,-1] )
        print( '---' )

    # Plot error results.
    fig, axs = plt.subplots( 1,2 )
    fig.suptitle( 'Formation Error' )
    ymax = np.max( eTrend[-1,:] )
    print( ymax )
    titles = ('Trend', 'Mean')
    for a, title in zip( axs, titles ):
        a.grid( 1 )
        a.set_ylim( [0, ymax+0.01] )
        a.set_title( title )
    s = round( 1.5/dt )  # Settling time used in mean.
    for eps, error in zip( epsList, eTrend ):
        label = '$\\varepsilon = %0.1f$' % eps
        eAvrg = np.mean( error )
        axs[0].plot( tList[0], error )
        axs[1].plot( [0, 1], [eAvrg, eAvrg], label=label )
    axs[1].legend()
    # fig.tight_layout()
    plt.show()
