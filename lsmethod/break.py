import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *


# Set hyper parameter(s).
# N = np.random.randint(1,10)
N = 3                    # Number of anchors.
M = N                    # Number of vehicles.

R = rotz( np.pi )
v = np.array(
    [[0, 0, 0],
    [1, 0, 0]] )

# Anchor set.
Aset = np.array( [
    [0, 3, -3],
    [2, 0, 0]] )
print( 'Aset:\n', Aset )


# For consistency with notes and error calc.
Xeq = Aset
PSI = lambda A: np.vstack( (A, np.ones( (1,A.shape[1]) )) )


# Calculate anchor coefficient matrices.
A, B = anchorDifferenceMatrices(Aset, N=M)
Z, _ = Regressor( A.T@A, np.eye( Nx,Nx ) ).dmd()
K = Z@A.T
print( 'A:', A )
print( 'B:', B )
print( 'K:', K )


# Main execution block.
if __name__ == '__main__':
    # Simulation time.
    T = 2;  Nt = round( T/dt ) + 1
    tList = np.array( [ [i*dt for i in range( Nt )] ] )

    # Initialize vehicle positions.
    delta = 5.0
    eps = 0.0
    X = R@(Xeq + v)

    # Initial error calculation.
    ge = 1
    regr = Regressor( PSI( Xeq ), X )
    T, _ = regr.dmd();
    e0 = np.vstack( (0, regr.err) )

    # Used for plotting without sim.
    xList = np.empty( (M,Nt,Nx) )
    eList = np.empty( (2,Nt) )
    xList[:,0,:] = X.T
    eList[:,0] = e0[:,0]

    # Initialize plot with vehicles, anchors and markers.
    fig, axs, swrm, anchors, error = initAnchorEnvironment(
        X, Xeq, Aset, e0, Nt=Nt, ge=1, R1=0.40, R2=delta, anchs=False, dist=False)

    # Environment block.
    print( 'Xi: %0.3f\n' % regr.err, X )
    input( "Press ENTER to begin simulation..." )
    for i in range( Nt ):
        # Calculate control term.
        U = asymmetricControl( X, Xeq, C, K, B )

        # Apply dynamics.
        X = model( X, U )

        # Calculate tranformation error.
        regr = Regressor( PSI( Xeq ), X )
        T, _ = regr.dmd()

        # Save values.
        eList[:,i] = np.array( [ge*i, regr.err] )
        xList[:,i,:] = X.T

        # Update simulation.
        if sim and i % n == 0:
            swrm.update( X )
            error.update( eList[:,i,None] )
            # axs[1].set_title( 'time: %s' % i )
            plt.pause( pausesim )
    print( 'Xf:\n', X )

    # Plot transformed grid for reference.
    finalAnchorEnvironment( fig, axs, swrm, xList, eList, T, shrink=0.1 )
    plt.pause( pausesim )

    # Calculate error after transformation.
    print( '\nError: ', eList[1,-1] )
    ans = input( "Press ENTER to exit program..." )
    if save or ans == 'save':
        fig.savefig( figurepath
            + 'ls_formation_d%i' % delta
            + '_e%i.png' % eps,
            dpi=600 )
        print( 'Figure saved.' )
