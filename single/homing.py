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
    eps = 0.0
    X0 = Xeq + delta/2 # noiseCirc( eps=delta, N=m )

    # Initialize simulation variables.
    e0 = np.vstack( ([0],anchoredLyapunovCandidate( X0, Xeq )) )
    fig, axs, swrm, anchors, error = initAnchorEnvironment(
        X0, Xeq, Aset, e0, Nt=Nt, delta=delta, anchs=True, dist=True )

    # Simulation block.
    X = X0;  e = e0
    for t in range( Nt ):
        # Anchor-based control.
        U = distanceBasedControl( X, Xeq, C, K, B, A=Aset, eps=eps )

        # Apply dynamics.
        X = model( X, U )
        e = np.vstack( ([t],anchoredLyapunovCandidate( X, Xeq )) )

        # Update simulation.
        if sim and t % n == 0:
            swrm.update( X )
            error.update( e )
            # axs[1].set_title( 'time: %s' % i )
            plt.pause( pausesim )

    input("Press ENTER to end program.")