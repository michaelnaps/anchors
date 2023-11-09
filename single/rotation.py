import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from plotfuncs import *


# Anchor initialization.
n = 3
m = 1
Aset = Abound/4*np.array( [
    [-1, 1, 1, -1],
    [1, 1, -1, -1] ] )

# For consistency with notes and error calc.
Xeq = np.array( [[0],[0]] )  # noiseCirc( eps=Abound/4, N=1 )

# Control formula components.
C, K, B = distanceBasedControlMatrices( Aset, m )

# Rotation list.
Nth = 3
thList = np.array( [k/Nth*2*np.pi for k in range( Nth+1 )] )
rotList = np.array( [rotz( theta ) for theta in thList] )

r0 = np.array( [[-Abound],[-Abound*Nth/2]] )
rList = 0*np.array( [[[0],[Nth*Abound*k/Nth]]+r0 for k in range( Nth+1 )] )

print( thList )
print( rList )

# Main execution block.
if __name__ == '__main__':
    # Time series variables.
    T = 10;  Nt = round( T/dt ) + 1
    tList = np.array( [[i*dt for i in range( Nt )]] )

    # Initial vehicle positions.
    X0 = Abound*np.kron( np.array( [[0],[1]] ), np.ones( (1,Nth+1) ) )
    V0 = np.vstack( ([0],lyapunovCandidateAnchored( X0, Xeq )) )

    # Used for plotting without sim.
    xList = np.nan*np.empty( (m*(Nth+1),Nt,Nx) )
    VList = np.nan*np.empty( (2,Nt) )
    xList[:,0,:] = X0.T
    VList[:,0] = V0[:,0]

    # Initialize simulation variables.
    fig, axs = plt.subplots( 1,Nth+1 )

    xswrmList = [None for i in range( Nth )]
    errorList = [None for i in range( Nth )]
    for i in range( Nth ):
        fig, axs[i], xswrmList[i], _, errorList[i] \
            = initAnchorEnvironment(
                X0[:,i,None], Xeq, rotList[i]@Aset + rList[i], V0,
                Nt=Nt, anchs=True, dist=False,
                fig=fig, axsX=axs[i], axsV=axs[-1] )
        # plotAnchors(fig, axs[0], R@Aset + r, radius=0.30,
        #     connect=True, color='orange')

    # Simulation block.
    X = X0
    for t in range( Nt-1 ):
        for i, (R, r) in enumerate( zip( rotList, rList ) ):
            # Get i-th vehicle positions.
            x = X[:,i,None]

            # Check if position is still within bounds.
            if np.linalg.norm( x ) > 3*Abound:
                break

            # Anchor-based control.
            u = distanceBasedControl( x, Xeq, C, K, B, A=R@Aset + r )[0]

            # Apply dynamics.
            X[:,i] = model( x, u )[:,0]

        # Lyapunov candidate function.
        V = np.vstack( ([t],lyapunovCandidateAnchored( X, Xeq )) )

        # Save values.
        xList[:,t+1,:] = X.T
        VList[:,t+1] = V[:,0]

    # Plot transformed grid for reference.
    for i in range( Nth ):
        finalAnchorEnvironmentAnchored( fig, (axs[i], axs[-1]), xswrmList[i],
            None, xList[i], None, VList[i], shrink=1 )

    axs[-1].set_ylabel( '$V(x)$' )
    legend_elements = [
        Line2D([0], [0], color='cornflowerblue', linestyle='none', marker='x',
            label='$X^{(0)}$'),
        Line2D([0], [0], color='cornflowerblue', marker='o', markerfacecolor='none',
            label='$X$'),
        Line2D([0], [0], color='indianred', linestyle='none', marker='o', markeredgecolor='k',
            label='$\\mathcal{A}$' ) ]
    axs[-1].legend( handles=legend_elements, ncol=1 )
    plt.pause( pausesim )

    input("Press ENTER to end program.")