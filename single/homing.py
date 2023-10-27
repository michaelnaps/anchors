import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *


# Anchor initialization.
n = 3
m = 1
Aset = noise( eps=Abound, shape=(2,n) )

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
    delta = Abound/2
    eps = 10.0
    X0 = Xeq + noiseCirc( eps=delta, N=m )
    e0 = np.vstack( ([0],anchoredLyapunovCandidate( X0, Xeq )) )

    # Used for plotting without sim.
    xList = np.nan*np.ones( (n,Nt,Nx) )
    eList = np.nan*np.ones( (2,Nt) )
    xList[:,0,:] = X0.T
    eList[:,0] = e0[:,0]

    # Initialize simulation variables.
    fig, axs, swrm, anchors, error = initAnchorEnvironment(
        X0, Xeq, Aset, e0, Nt=Nt, delta=delta, anchs=True, dist=True )

    # Simulation block.
    X = X0;  e = e0
    for t in range( Nt ):
        # Anchor-based control.
        U = distanceBasedControl( X, Xeq, C, K, B, A=Aset, eps=eps )

        # Apply dynamics.
        X = model( X, U )
        V = np.vstack( ([t],anchoredLyapunovCandidate( X, Xeq )) )

        # Save values.
        eList[:,t] = V[:,0]
        xList[:,t,:] = X.T

        # Update simulation.
        if sim and t % n == 0:
            swrm.update( X )
            error.update( V )
            plt.pause( pausesim )

    # Plot transformed grid for reference.
    finalAnchorEnvironmentAnchored( fig, axs, swrm, xList, eList, shrink=1 )
    plt.pause( pausesim )

    input("Press ENTER to end program.")