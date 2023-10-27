import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *


# Anchor initialization.
n = 3
m = 10
Aset = Abound/2*np.array( [
    [-1, 1, 1],
    [1, 1, -1] ] )

# For consistency with notes and error calc.
Xeq = noiseCirc( eps=Abound, N=1 )

# Control formula components.
C, K, B = distanceBasedControlMatrices( Aset, m )

# Main execution block.
if __name__ == '__main__':
    # Time series variables.
    T = 1;  Nt = round( T/dt ) + 1
    tList = np.array( [[i*dt for i in range( Nt )]] )

    # Initial vehicle positions.
    delta = Abound
    eps = 0.5
    X0 = Xeq + noiseCirc( eps=delta, N=m )
    e0 = np.vstack( ([0],lyapunovCandidateAnchored( X0, Xeq )) )

    # Used for plotting without sim.
    xList = np.nan*np.ones( (m,Nt,Nx) )
    yList = np.nan*np.ones( (m,Nt,Nx) )
    eList = np.nan*np.ones( (2,Nt) )
    xList[:,0,:] = X0.T
    yList[:,0,:] = X0.T
    eList[:,0] = e0[:,0]

    # Initialize simulation variables.
    fig, axs, xswrm, anchors, error = initAnchorEnvironment(
        X0, Xeq, Aset, e0, Nt=Nt, delta=delta, anchs=True, dist=False )
    yswrm = Swarm2D( X0, fig=fig, axs=axs[0], zorder=200,
        radius=-0.30, color='yellowgreen', tail_length=Nt,
        draw_tail=sim ).draw()

    # Simulation block.
    X = X0;  Y = X0;  e = e0
    for t in range( Nt ):
        # Anchor-based control.
        X = Y + noiseCirc( eps=eps, N=m )
        U = distanceBasedControl( X, Xeq, C, K, B, A=Aset )

        # Apply dynamics.
        Y = model( Y, U )
        V = np.vstack( ([t],lyapunovCandidateAnchored( X, Xeq )) )

        # Save values.
        xList[:,t,:] = X.T
        yList[:,t,:] = Y.T
        eList[:,t] = V[:,0]

        # Update simulation.
        if sim and t % n == 0:
            xswrm.update( X )
            yswrm.update( Y )
            error.update( V )
            plt.pause( pausesim )

    # Plot transformed grid for reference.
    finalAnchorEnvironmentAnchored( fig, axs, xswrm, yswrm, xList, yList, eList, shrink=1 )
    axs[1].set_ylabel( '$V(x)$' )
    legend_elements = [
        Line2D([0], [0], color='cornflowerblue', linestyle='none', marker='x',
            label='$X^{(0)}$'),
        Line2D([0], [0], color='yellowgreen', marker='o', markerfacecolor='none',
            label='$X$'),
        Line2D([0], [0], color='cornflowerblue', marker='o', markerfacecolor='none',
            label='$X + p(\\varepsilon)$'),
        Line2D([0], [0], color='indianred', linestyle='none', marker='o', markeredgecolor='k',
            label='$\\mathcal{A}$' )
    ]
    axs[1].legend( handles=legend_elements, ncol=1 )
    plt.pause( pausesim )

    input("Press ENTER to end program.")