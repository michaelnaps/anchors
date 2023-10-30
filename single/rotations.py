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
Nth = 40
thList = np.array( [k/Nth*2*np.pi for k in range( Nth+1 )] )
rotList = np.array( [rotz( theta ) for theta in thList] )
# rotList = np.array( [noise( eps=100, shape=(2,m) ) for k in range( Nth+1 )] )

# Main execution block.
if __name__ == '__main__':
    # Time series variables.
    T = 2;  Nt = round( T/dt ) + 1
    tList = np.array( [[i*dt for i in range( Nt )]] )

    # Initial vehicle positions.
    X0 = Abound*np.kron( np.array( [[0],[1]] ), np.ones( (1,Nth+1) ) )
    e0 = np.vstack( ([0],lyapunovCandidateAnchored( X0, Xeq )) )

    # Used for plotting without sim.
    xList = np.nan*np.ones( (m*(Nth+1),Nt,Nx) )
    eList = np.nan*np.ones( (2,Nt) )
    xList[:,0,:] = X0.T
    eList[:,0] = e0[:,0]

    # Initialize simulation variables.
    fig, axs, xswrm, anchors, error = initAnchorEnvironment(
        X0, Xeq, Aset, e0, Nt=Nt, anchs=True, dist=False )

    # Simulation block.
    X = X0;  e = e0
    for t in range( Nt-1 ):
        for i, R in enumerate( rotList ):
            # Get i-th vehicle positions.
            x = X[:,i,None]

            # Check if position is still within bounds.
            if np.linalg.norm( x ) > 2*Abound:
                break

            # Anchor-based control.
            u = distanceBasedControl( x, Xeq, C, K, B, A=R@Aset )[0]

            # Apply dynamics.
            X[:,i] = model( x, u )[:,0]

        # Lyapunov candidate function.
        V = np.vstack( ([t],lyapunovCandidateAnchored( X, Xeq )) )

        # Save values.
        xList[:,t+1,:] = X.T
        eList[:,t] = V[:,0]

        # Update simulation.
        if sim and t % n == 0:
            xswrm.update( X )
            error.update( V )
            plt.pause( pausesim )

    # Plot transformed grid for reference.
    finalAnchorEnvironmentAnchored( fig, axs, xswrm, None, xList, None, eList, shrink=1 )
    axs[1].set_ylabel( '$V(x)$' )
    legend_elements = [
        Line2D([0], [0], color='cornflowerblue', linestyle='none', marker='x',
            label='$X^{(0)}$'),
        Line2D([0], [0], color='cornflowerblue', marker='o', markerfacecolor='none',
            label='$X$'),
        Line2D([0], [0], color='indianred', linestyle='none', marker='o', markeredgecolor='k',
            label='$\\mathcal{A}$' ) ]
    axs[1].legend( handles=legend_elements, ncol=1 )
    plt.pause( pausesim )

    input("Press ENTER to end program.")