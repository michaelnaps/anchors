import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *


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
rList = 2*Abound*np.array(
    [rotz(k*2*np.pi/(Nr))@[[1],[0]] for k in range( Nr )] )

# Main execution block.
if __name__ == '__main__':
    # Time series variables.
    T = 1;  Nt = round( T/dt ) + 1
    tList = np.array( [[i*dt for i in range( Nt )]] )

    # Initial vehicle positions.
    X0 = np.zeros( (Nx,Nr) )
    V0 = np.vstack( ([0],lyapunovCandidateAnchored( X0, Xeq )) )

    # Used for plotting without sim.
    xList = np.nan*np.ones( (m*(Nr),Nt,Nx) )
    VList = np.nan*np.ones( (2,Nt) )
    xList[:,0,:] = X0.T
    VList[:,0] = V0[:,0]

    # Initialize simulation variables.
    fig, axs, xswrm, anchors, error = initAnchorEnvironment(
        X0, Xeq, Aset, V0, Nt=Nt, radius=1.00, anchs=True, dist=False )
    for r in rList:
        plotAnchors(fig, axs[0], Aset + r, radius=0.85,
            connect=True, color='orange')

    # Simulation block.
    X = X0;
    for t in range( Nt-1 ):
        V = np.array( [[t],[0]] )
        for i, r in enumerate( rList ):
            # Get i-th vehicle positions.
            x = X[:,i,None]

            # Check if position is still within bounds.
            if np.linalg.norm( x ) > 2*Abound:
                break

            # Anchor-based control.
            u = distanceBasedControl( x, Xeq, C, K, B, A=Aset + r )[0]

            # Apply dynamics.
            X[:,i] = model( x, u )[:,0]
            V[1] = V[1] + lyapunovCandidateAnchored( x, Xeq, r=r )

        # Save values.
        xList[:,t+1,:] = X.T
        VList[:,t] = V[:,0]

        # Update simulation.
        if sim and t % n == 0:
            xswrm.update( X )
            error.update( V )
            plt.pause( pausesim )

    # Plot transformed grid for reference.
    finalAnchorEnvironmentAnchored( fig, axs, xswrm, None, xList, None, VList, shrink=1 )
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