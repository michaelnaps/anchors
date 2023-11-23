import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from plotfuncs import *


# Anchor set.
delta = 2.0
Aset = Abound/2*np.array( [
    [-1,  0,  1,  0,  1],
    [ 0,  0,  0,  1,  1]] )

# Set dimensions
n = Aset.shape[1]        # Number of anchors.
m = n                    # Number of vehicles.

# For consistency with notes and error calc.
Xeq = Aset

# Control formula components.
C, K, B = distanceBasedControlMatrices( Aset, m )

# Main execution block.
if __name__ == '__main__':
    # Time series variables.
    T = 10.0;  Nt = round( T/dt ) + 1
    tList = np.array( [[i*dt for i in range( Nt )]] )

    # Set parameters.
    RList = [ rotz( np.pi ) ]
    Nr = len( RList )
    colorList = ['cornflowerblue', 'yellowgreen']

    # Initial vehicle positions.
    eps = 0.1
    X0 = Xeq + noiseCirc( eps=eps, N=m )
    V0 = np.zeros( (2,1) )

    # Used for plotting without sim.
    xList = np.nan*np.ones( (Nr,m,Nt,Nx) )
    yList = np.nan*np.ones( (Nr,m,Nt,Nx) )
    VList = np.nan*np.ones( (Nr,1,Nt,2) )

    # Initialize simulation variables.
    fig, axs = plt.subplots(1,2)

    # Simulation block.
    xswrm = [None for i in range( Nr )]
    yswrm = [None for i in range( Nr )]
    cand = [None for i in range( Nr )]
    for i, R in enumerate( RList ):
        X = R@X0
        Y = distanceBasedControl( X, Xeq, C, K, B )[1]
        V = np.hstack( ([0], lyapunovCandidate( X, Xeq )[0]) )

        _, _, xswrm[i], _, _ = initEnvironment(
            fig, axs, X, Xeq, Aset, V0,
            color=colorList[i], Nt=Nt, anchs=True )
        # yswrm[i] = Swarm2D( Y, fig=fig, axs=axs[i], zorder=z_swrm-100,
        #     radius=-0.30, color='yellowgreen', tail_length=Nt,
        #     draw_tail=sim ).setLineStyle( '--' ).draw()

        xList[i,:,0] = X.T
        yList[i,:,0] = Y.T
        VList[i,:,0] = V.T
        for t in range( Nt-1 ):
            # Anchor-based control.
            U, Y = distanceBasedControl( X, Xeq, C, K, B )

            # Apply dynamics.
            X = model( X, U )

            # Lyapunov function.
            V = lyapunovCandidate( X, Xeq )

            # Save values.
            xList[i,:,t+1] = X.T
            yList[i,:,t+1] = Y.T
            VList[i,:,t+1] = np.hstack( ([t+1], V[0]) )

            # Check for convergence/divergence of Lyapunov candidate.
            if V > 250:
                xList[i,:,-1] = X.T
                print( 'Formation policy diverged.' )
                break

        plotEnvironment( fig, axs, xswrm[i], xList[i], VList[i],
            xcolor=colorList[i], plotXf=True )
        # plotEnvironment( fig, [axs[i], axs[-1]], yswrm[i], yList[i],
        #     plotXf=True, zorder=z_swrm-100 )

    # # Plot final anchor positions.
    # abar = centroid( Aset )
    # for xf in xList[:,:,-1]:
    #     X = xf.T;  xbar = centroid( X )
    #     Psi = rotation( Aset-abar, X-xbar )
    #     psi = xbar - Psi@abar
    #     plotAnchors( fig, axs, Psi@Aset + psi, anchs=False, zdelta=400 )


    # Plot and axis labels.
    titles = ['$\\theta = %s$' % th for th in ['\\pi/2']] + ['Formation Error']
    xlabels = Nr*['x'] + ['Iteration']
    ylabels = ['$y$'] + (Nr-1)*[None] + ['$W(X)$']
    for a, title, xlabel, ylabel in zip( axs, titles, xlabels, ylabels ):
        a.set_title( title )
        a.set_xlabel( xlabel )
        a.set_ylabel( ylabel )

    # Plot transformed grid for reference.
    legend_elements_1 = [
        Line2D([0], [0], color='cornflowerblue', linestyle='none', marker='x',
            label='$X^{(0)}$'),
        Line2D([0], [0], color='cornflowerblue', marker='o', markerfacecolor='none',
            label='$X$'),
        Line2D([0], [0], color='indianred', linestyle='none', marker='o', markeredgecolor='k',
            label='$\\mathcal{A}, X^{\\textrm{(eq)}}$' ),
    ]
    axs[1].legend( handles=legend_elements_1, fontsize=fontsize-2, ncol=1, loc='upper left' )

    scale = 1
    fig.set_figwidth( scale*plt.rcParams.get('figure.figsize')[0] )
    fig.set_figheight( 4/5*scale*figheight )
    fig.tight_layout()
    if show:
        plt.show( block=0 )

    # Calculate error after transformation.
    ans = input( 'Press ENTER to exit program... ' )
    if save or ans == 'save':
        filename = 'multi/spin.pdf'
        fig.savefig( figurepath + filename, dpi=600 )
        print( 'Figure saved to:\n ' + figurepath + filename )