import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from plotfuncs import *


# Set hyper parameter(s).
# N = np.random.randint(1,10)
n = 16                   # Number of anchors.
m = n                    # Number of vehicles.

# Anchor set.
delta = 2.0
Aset = np.hstack( (
    [rotz( 2*np.pi*k/n - np.pi/2 )@[[k/2],[0]] for k in range( 1,n+1 )] ) )

# For consistency with notes and error calc.
Xeq = Aset

# Control formula components.
C, K, B = distanceBasedControlMatrices( Aset, m )

# Main execution block.
if __name__ == '__main__':
    # Time series variables.
    T = 2.5;  Nt = round( T/dt ) + 1
    tList = np.array( [[i*dt for i in range( Nt )]] )

    # Set parameters.
    Ne = 3
    epsList = [1.0, 5.0, 10.0]
    vcolor = ['mediumpurple', 'cornflowerblue', 'mediumseagreen']
    vlinestyle = ['solid' for _ in range( Ne )]

    # Initial vehicle positions.
    X0 = Xeq
    V0 = np.zeros( (2,1) )

    # Used for plotting without sim.
    xList = np.nan*np.ones( (m,Nt,Nx) )
    yList = np.nan*np.ones( (m,Nt,Nx) )
    VList = np.nan*np.ones( (1,Nt,2) )

    # Initialize simulation variables.
    fig, axs = plt.subplots( 1,Ne+1 )
    fig.set_figwidth( 3.0*plt.rcParams.get('figure.figsize')[0] )

    # Simulation block.
    xswrm = [None for i in range( Ne )]
    yswrm = [None for i in range( Ne )]
    cand = [None for i in range( Ne )]
    for i in range( Ne ):
        _, _, xswrm[i], _, _ = initEnvironment(
            fig, [axs[i], axs[-1]], X0, Xeq, Aset, V0, Nt=Nt, anchs=False )
        yswrm[i] = Swarm2D( X0, fig=fig, axs=axs[i], zorder=z_swrm-100,
            radius=-0.30, color='yellowgreen', tail_length=Nt,
            draw_tail=sim ).setLineStyle( '--' ).draw()

    # Simulation block.
    for i, eps in enumerate( epsList ):
        X = X0 + noiseCirc( eps=eps, N=m )
        Y = X
        V0 = lyapunovCandidate( X0, Xeq )

        xList[:,0,:] = X0.T
        yList[:,0,:] = X0.T
        VList[:,0,:] = np.hstack( ([0], V0[0]) )
        for t in range( Nt-1 ):
            # Anchor-based control.
            U, Y = distanceBasedControl( X, Xeq, C, K, B )

            # Apply dynamics.
            X = model( X, U )

            # Lyapunov function.
            V = lyapunovCandidate( X, Xeq )

            # Save values.
            xList[:,t+1,:] = X.T
            yList[:,t+1,:] = Y.T
            VList[:,t+1,:] = np.hstack( ([t+1], V[0]) )

            # Check for convergence/divergence of Lyapunov candidate.
            if V > 1e3:
                break
            # elif V < 1e-24:
            #     break

        plotEnvironment( fig, [axs[i], axs[-1]], xswrm[i], xList, VList,
            plotXf=False, color=vcolor[i], linestyle=vlinestyle[i] )
        plotEnvironment( fig, [axs[i], axs[-1]], yswrm[i], yList,
            plotXf=False, zorder=z_swrm-100 )

    # Plot and axis labels.
    titles = ['$\\varepsilon = %.1f$' % eps for eps in epsList] + ['Lyapunov Trend']
    xlabels = [None, '$x$', None, 'Iteration']
    ylabels = ['$y$', None, None, '$V(x)$']
    for a, title, xlabel, ylabel in zip( axs, titles, xlabels, ylabels ):
        a.set_title( title )
        a.set_xlabel( xlabel )
        a.set_ylabel( ylabel )

    # Plot transformed grid for reference.
    legend_elements_1 = [
        Line2D([0], [0], color='cornflowerblue', linestyle='none', marker='x',
            label='$X^{(0)}$'),
        Line2D([0], [0], color='indianred', linestyle='none', marker='o', markeredgecolor='k',
            label='$\\mathcal{A}$' ),
        Line2D([0], [0], color='cornflowerblue', marker='o', markerfacecolor='none',
            label='$X$'),
        Line2D([0], [0], color='yellowgreen', linewidth=1, marker='o', markerfacecolor='none',
            label='$K(h(x) - b)$'),
    ]
    axs[0].axis( Abound*np.array( [-1, 1, -1.1, 1.2] ) )
    axs[0].legend( handles=legend_elements_1, ncol=2, loc=1 )

    legend_elements_2 = [
        Line2D([0], [0], color=vcolor[i], linestyle=vlinestyle[i], linewidth=2,
            label='$\\varepsilon = %.1f$' % eps) for i, eps in enumerate( epsList )
    ]
    axs[-1].legend( handles=legend_elements_2, ncol=1 )

    fig.tight_layout()
    plt.pause( pausesim )

    # Calculate error after transformation.
    ans = input( 'Press ENTER to exit program... ' )
    if save or ans == 'save':
        filename = 'single/homing.png'
        fig.savefig( figurepath + filename, dpi=600 )
        print( 'Figure saved to:\n ' + figurepath + filename )