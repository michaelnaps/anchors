import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from plotfuncs import *


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
    V0 = np.vstack( (0, lyapunovCandidate( X, Xeq )) )

    # Used for plotting without sim.
    xList = np.nan*np.ones( (M,Nt,Nx) )
    VList = np.nan*np.ones( (1,Nt,2) )
    xList[:,0,:] = X.T
    VList[:,0,:] = V0.T

    # Initialize plot with vehicles, anchors and markers.
    fig, axs, swrm, anchors, error = initAnchorEnvironment(
        X, Xeq, Aset, V0, Nt=Nt, radius=0.40, delta=delta,
        anchs=False, dist=False )

    # Environment block.
    for i in range( Nt ):
        # Calculate control term.
        U = distanceBasedControl( X, Xeq, C, K, B, eps=eps )[0]

        # Apply dynamics.
        X = model( X, U )

        # Calculate tranformation error.
        V = lyapunovCandidate( X, Xeq )

        # Save values.
        VList[:,i,:] = np.array( [ge*i, V[0][0]] )
        xList[:,i,:] = X.T

    # Tranformation.
    Xbar = centroid( X )
    Abar = centroid( Xeq )
    Psi = rotation( X - Xbar, Xeq - Abar )

    finalAnchorEnvironment( fig, axs, swrm, xList, VList, Psi, Xbar, shrink=3/4 )

    # Legend setup.
    axs[0].set_ylabel( '$V(x)$' )
    legend_elements = [
        Line2D([0], [0], color='cornflowerblue', linestyle='none', marker='x',
            label='$X^{(0)}$'),
        Line2D([0], [0], color='cornflowerblue', marker='o', markerfacecolor='none',
            label='$X$'),
        Line2D([0], [0], color='indianred', linestyle='none', marker='o', markeredgecolor='k',
            label='$\\mathcal{A}, X^{(eq)}$' ),
        Line2D([0], [0], color='grey', linestyle='--', marker='o', markerfacecolor='grey',
            label='$\Psi [x,y]^\\top + \psi$') ]
    axs[-1].legend( handles=legend_elements, ncol=1 )
    plt.pause( pausesim )

    # Calculate error after transformation.
    ans = input( 'Press ENTER to exit program... ' )
    if save or ans == 'save':
        fig.savefig( figurepath
            + 'multi/formation_d%i' % delta + '.png' % eps,
            dpi=600 )
        print( 'Figure saved.' )
