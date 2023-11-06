import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from plotfuncs import *


# Anchor initialization.
n = 3
m = 1
Aset = Abound/2*np.array( [
    [-1, 1, 1, -1],
    [1, 1, -1, -1] ] )

# For consistency with notes and error calc.
Xeq = np.array( [[0],[0]] )  # noiseCirc( eps=Abound/4, N=1 )

# Control formula components.
C, K, B = distanceBasedControlMatrices( Aset, m )

# Rotation list.
Nr = 3
R = rotz( 0.0 )
rList = 2*Abound*np.array(
    [rotz(k*2*np.pi/(Nr))@[[1],[0]] for k in range( Nr )] )

# Main execution block.
if __name__ == '__main__':
    # Time series variables.
    T = 5;  Nt = round( T/dt ) + 1
    tList = np.array( [[i*dt for i in range( Nt )]] )

    # Initial vehicle positions.
    X0 = np.zeros( (Nx,Nr) )
    V0 = np.zeros( (2,1) )

    # Used for plotting without sim.
    xList = np.nan*np.ones( (m*Nr,Nt,Nx) )
    VList = np.nan*np.ones( (m*Nr,Nt,2) )
    xList[:,0,:] = X0.T
    VList[:,0,:] = lyapunovCandidatePerVehicle( m*Nr, 0, X0, Xeq ).T

    # Initialize simulation variables.
    fig, axs = plt.subplots( 1,2 )
    fig, axs, xswrm, anchors, cand = initEnvironment(
        fig, axs, X0, Xeq, Aset, V0, Nt=Nt, radius=1, connect=True)
    for r in rList:
        plotAnchors( fig, axs[0], R@Aset + r, radius=0.85,
            color='orange', connect=True )

    # Simulation block.
    X = X0
    for t in range( Nt-1 ):
        for i, r in enumerate( rList ):
            # Get i-th vehicle positions.
            x = X[:,i,None]

            # Check if position is still within bounds.
            if np.linalg.norm( x ) > 10*Abound:
                break

            # Anchor-based control.
            u = distanceBasedControl( x, Xeq, C, K, B, A=R@Aset + r )[0]

            # Apply dynamics.
            X[:,i] = model( x, u )[:,0]
            VList[i,t+1,:] = lyapunovCandidatePerVehicle( m, t+1, x, R@Xeq+r ).T

        # Save values.
        xList[:,t+1,:] = X.T
        # VList[:,t+1,:] = V.T
        print( VList )

        # Update simulation.
        if sim and t % n == 0:
            xswrm.update( X )
            cand.update( V )
            plt.pause( pausesim )

    # Plot transformed grid for reference.
    fig, axs = plotEnvironment( fig, axs, xswrm, xList, VList )
    axs[1].set_ylabel( '$V(x)$' )
    legend_elements = [
        Line2D([0], [0], color='cornflowerblue', linestyle='none', marker='x',
            label='$X^{(0)}$'),
        Line2D([0], [0], color='cornflowerblue', marker='o', markerfacecolor='none',
            label='$X$'),
        Line2D([0], [0], color='indianred', linestyle='none', marker='o', markeredgecolor='k',
            label='$\\mathcal{A}$' ),
        Line2D([0], [0], color='orange', linestyle='none', marker='o', markeredgecolor='k',
            label='$\\mathcal{A} + r$' ) ]
    axs[1].legend( handles=legend_elements, ncol=1 )
    plt.pause( pausesim )

    # Calculate error after transformation.
    ans = input( 'Press ENTER to exit program... ' )
    if save or ans == 'save':
        fig.savefig( figurepath + 'single/offset_r%i.png' % (Nr), dpi=600 )
        print( 'Figure saved.' )