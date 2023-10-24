import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *


# I like these points.
# [[ 2.545  1.373 -0.815 -4.318 -2.402]
#  [ 4.469 -3.036  2.047  1.842 -3.984]]
# [[ 2.172  0.397 -2.094 -2.668]
#  [ 4.347  0.989  3.068 -8.288]]


# Set hyper parameter(s).
Nr = 3                      # Number of anchor sets + reflection sets.
N = 1                       # Number of anchors.
M = N*Nr                    # Number of symmetric vehicles.
P = M + 1                   # Number of vehicles.


# Exclusion elements in measurement function.
def exclude(i, j):
    return i > M - 1


# Anchor set.
A = np.array( [[2],[3]] )
# A = Abound*np.random.rand( 2,N )
print( 'A:\n', A )

# Asymmetric vehicle set.
B = np.array( [[-3],[-7]] )
# B = -Abound*np.random.rand( 2,P-M )

# Reflection sets.
Ax = Rx@A
Ay = Ry@A
Q = np.hstack( (A, Ax, Ay, B) )
print( 'Ax:\n', Ax )
print( 'Ay:\n', Ay )
print( 'Q:\n', Q )

# For error calculation.
Qerr = np.vstack( (Q, np.ones( (1,P) )) )

# Herd leader.
W = 10.0
p = 1

# Calculate anchor coefficient matrices.
S = signedCoefficientMatrix( N )
Z = anchorCoefficientMatrix( A, N, exclude=exclude )
print( 'S:\n', S )
print( 'Z:\n', Z )


# Main execution block.
if __name__ == '__main__':
    # Simulation time.
    T = 5;  Nt = round( T/dt ) + 1
    tList = np.array( [ [i*dt for i in range( Nt )] ] )

    # Initialize vehicle positions.
    delta = 5.0
    eps = 0.0
    X = Q + noiseCirc( eps=delta, N=P )

    # Initial error calculation.
    ge = 1
    regr = Regressor( Qerr, X )
    T, _ = regr.dmd()
    e0 = np.vstack( (0, regr.err) )

    # Used for plotting without sim.
    xList = np.empty( (P,Nt,Nx) )
    eList = np.empty( (2,Nt) )
    xList[:,0,:] = X.T
    eList[:,0] = e0[:,0]

    # Initialize plot with vehicles, anchors and markers.
    fig, axs, swrm, anchors, error = initAnchorEnvironment(
        X[:,1:], Q[:,1:], A[:,1:], e0, Nt=Nt, ge=1, R1=0.40, R2=delta, anchs=False, dist=False)
    leader = Vehicle2D( X[:,0,None], fig=fig, axs=axs[0],
        radius=-0.40, tail_length=Nt, color='yellowgreen' ).draw()
    axs[0].axis( 1.5*Abound*np.array( [-1, 2, -1, 2] ) )

    # Environment block.
    print( 'Xi: %0.3f\n' % regr.err, X )
    input( "Press ENTER to begin simulation..." )
    for i in range( Nt ):
        # Take measurements.
        H = anchorMeasure( X, X, eps=eps, exclude=exclude )**2

        # Calculate control and add disturbance.
        U = C@(Q - (Z*S)@H[:N*Nr])
        if i > 200 and i < 225:
            U[:,:p] = W*np.ones( (Nx,p) )
        else:
            U[:,:p] = np.zeros( (Nx,p) )

        # Apply dynamics.
        X = model( X, U )

        # Calculate tranformation error.
        regr = Regressor( Qerr, X )
        T, _ = regr.dmd()

        # Save values.
        eList[:,i] = np.array( [i, regr.err] )
        xList[:,i,:] = X.T

        # Update simulation.
        if sim and i % n == 0:
            swrm.update( X[:,1:] )
            leader.update( X[:,0,None] )
            error.update( eList[:,i,None] )
            plt.pause( pausesim )
    print( 'Xf:\n', X )

    # Calculate transformation matrix by DMD.
    Qerr = np.vstack( (Q, np.ones( (1,P) )) )
    regr = Regressor( Qerr, X )
    T, _ = regr.dmd()
    print( 'T:\n', T )

    # Plot transformed grid for reference.
    finalAnchorEnvironment( fig, axs, swrm, xList, eList, T, shrink=1/3 )
    plt.pause( pausesim )

    # Calculate error after transformation.
    print( '\nError: ', np.linalg.norm( X - T@Qerr ) )
    input( "Press ENTER to exit program..." )

    if save:
        fig.savefig( figurepath
            + 'asymmetric_formation_d%i' % delta
            + '_e%i.png' % eps,
            dpi=600 )
        print( 'Figure saved.' )
