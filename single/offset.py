import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from plotfuncs import *


# Anchor initialization.
n = 4
m = 1
delta = 3.0
Aset = Abound/2*np.array( [
    [-1, 1, 1, -1],
    [1, 1, -1, -1] ]
) + noiseCirc( eps=delta, N=n )

# For consistency with notes and error calc.
Xeq = np.array( [[0],[0]] )  # noiseCirc( eps=Abound/4, N=1 )

# Control formula components.
C, K, B = distanceBasedControlMatrices( Aset, m )

# Rotation list.
Nr = 3
R = rotz( 0.0 )
rList = 2*Abound*np.array(
    [rotz(k*2*np.pi/(Nr))@[[1],[1/4]] for k in range( Nr )] )

# Main execution block.
if __name__ == '__main__':
    # Time series variables.
    T = 1;  Nt = round( T/dt ) + 1
    tList = np.array( [[i*dt for i in range( Nt )]] )

    # Initial vehicle positions.
    X0 = np.zeros( (Nx,Nr) )
    V0 = np.zeros( (2,1) )

    # Used for plotting without sim.
    xList = np.nan*np.ones( (m*Nr,Nt,Nx) )
    VList = np.nan*np.ones( (m*Nr,Nt,2) )
    xList[:,0,:] = X0.T

    # Initialize simulation variables.
    fig, axs = plt.subplots()
    _, _, xswrm, _, _ = initEnvironment(
        fig, [axs, None], X0, Xeq, Aset, V0, Nt=Nt, radius=1, connect=True)
    for r in rList:
        plotAnchors( fig, axs, R@Aset + r, color='orange', radius=1, connect=True )

    # Simulation block.
    for i, r in enumerate( rList ):
        # Get i-th vehicle positions.
        x = X0[:,i,None]
        xeq = Xeq
        VList[i,0,:] = [0, lyapunovCandidateAnchored( x, xeq, r=r )[0][0]]
        for t in range( Nt-1 ):
            # Anchor-based control.
            u = distanceBasedControl( x, xeq, C, K, B, A=R@Aset + r )[0]

            # Apply dynamics.
            x = model( x, u )

            # Save values.
            VList[i,t+1,:] = [t+1, lyapunovCandidateAnchored( x, xeq, r=r )[0][0]]
            xList[i,t+1,:] = x[:,0]

    # Plot transformed grid for reference.
    plotEnvironment( fig, [axs, None], xswrm, xList )
    legend_elements = [
        Line2D([0], [0], color='cornflowerblue', linestyle='none', marker='x',
            label='$X^{(0)}$'),
        Line2D([0], [0], color='cornflowerblue', marker='o', markerfacecolor='none',
            label='$X$'),
        Line2D([0], [0], color='indianred', marker='x', linestyle='none',
            label='$X^{(\\text{eq})}$'),
        Line2D([0], [0], color='indianred', linestyle='none', marker='o', markeredgecolor='k',
            label='$\\mathcal{A}$' ),
        Line2D([0], [0], color='orange', linestyle='none', marker='o', markeredgecolor='k',
            label='$R\\mathcal{A} + r$' ),
    ]
    axs.legend( handles=legend_elements, fontsize=fontsize-2, ncol=1, loc='lower right' )

    # Axes labels.
    axs.set_title( 'Variations in $r$' )
    axs.set_xlabel( '$x$' )
    axs.set_ylabel( '$y$' )
    # axs[1].set_xlabel( 'Iteration' )
    # axs[1].set_ylabel( '$V(x)$' )

    # Figure dimensions.
    fig.set_figwidth( 3/5*plt.rcParams.get('figure.figsize')[0] )
    fig.set_figheight( 4/5*figheight )
    fig.tight_layout()
    if show:
        plt.show( block=0 )

    # Calculate error after transformation.
    ans = input( 'Press ENTER to exit program... ' )
    if save or ans == 'save':
        fig.savefig( figurepath + 'single/offset_test.pdf', dpi=600 )
        print( 'Figure saved.' )