import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *


# Set hyper parameter(s).
# N = np.random.randint(1,10)
N = 16                   # Number of anchors.
M = N                    # Number of vehicles.

# Anchor set.
# Aset = Abound/2*np.array( [[-1,1,1],[1,1,-1]] )
# Aset = noiseCirc( eps=Abound, N=N )
# Aset = noise( eps=Abound, shape=(2,N) )
# Aset = np.array( [        # SMILEY FACE
#     [-3, -3, -3, -3, 3, 3, 3, 3, -5, -3.5, -2, -0.5, 0.5, 2, 3.5, 5],
#     [2, 4, 6, 8, 2, 4, 6, 8, 0, -1.5, -3, -3.5, -3.5, -3, -1.5, 0] ] )
Aset = np.hstack( (
    [rotz( 2*np.pi*k/N - np.pi/2 )@[[k/2],[0]] for k in range( 1,N+1 )] ) )
print( 'Aset:\n', Aset )

# For consistency with notes and error calc.
Xeq = Aset

# Control formula components.
C, K, B = distanceBasedControlMatrices( Aset, M )

# Main execution block.
if __name__ == '__main__':
    # Simulation time.
    T = 10;  Nt = round( T/dt ) + 1
    tList = np.array( [ [i*dt for i in range( Nt )] ] )

    # Initialize vehicle positions.
    delta = 10.0
    eps = 0.0
    X = Xeq + noiseCirc( eps=delta, N=M )

    # Initial error calculation.
    ge = 1
    e0 = np.vstack( (0, lyapunovCandidate( X, Xeq )) )

    # Used for plotting without sim.
    xList = np.nan*np.ones( (M,Nt,Nx) )
    eList = np.nan*np.ones( (2,Nt) )
    xList[:,0,:] = X.T
    eList[:,0] = e0[:,0]

    # Initialize plot with vehicles, anchors and markers.
    fig, axs, swrm, anchors, error = initAnchorEnvironment(
        X, Xeq, Aset, e0, Nt=Nt, ge=1, R1=0.40, R2=delta, anchs=False, dist=False )

    # Environment block.
    print( 'Xi: %0.3f\n' % e0[1,0], X )
    input( "Press ENTER to begin simulation..." )
    for i in range( Nt ):
        # Calculate control term.
        U = distanceBasedControl( X, Xeq, C, K, B, eps=eps )[0]

        # Apply dynamics.
        X = model( X, U )

        # Calculate tranformation error.
        V = lyapunovCandidate( X, Xeq )

        # Save values.
        eList[:,i] = np.array( [ge*i, V] )
        xList[:,i,:] = X.T

        # Update simulation.
        if sim and i % n == 0:
            swrm.update( X )
            error.update( eList[:,i,None] )
            # axs[1].set_title( 'time: %s' % i )
            plt.pause( pausesim )

        # Equilibrium break.
        if V < 1e-6:
            for j in range( 1,Nt-i ):
                xList[:,i+j,:] = xList[:,i,:]
            print( 'Equilibrium reached: V(X) = %.3e' % V )
            break
    print( 'Xf:\n', X )

    # Plot transformed grid for reference.
    Xbar = centroid( X )
    Abar = centroid( Xeq )

    Psi, _ = Regressor( X - Xbar, Xeq - Abar ).dmd()
    finalAnchorEnvironment( fig, axs, swrm, xList, eList, Psi, Xbar, shrink=1 )
    plt.pause( pausesim )

    # Calculate error after transformation.
    print( '\nError: ', eList[1,np.isfinite(eList[1])][-1] )
    ans = input( 'Press ENTER to exit program... ' )
    if save or ans == 'save':
        fig.savefig( figurepath
            + 'ls_formation_d%i' % delta + '_e%i.png' % eps,
            dpi=600 )
        print( 'Figure saved.' )
