import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from plotfuncs import *


# Anchor/vehicle dimensions.
n = 3
m = 10

# Anchor position definition.
delta = 2.0
Aset = Abound/2*np.array( [
    [0, -1, 1],
    [1, -1, -1] ]
) + noiseCirc( eps=delta, N=n )

# For consistency with notes and error calc.
Xeq = np.array( [[0],[0]] )  # noiseCirc( eps=Abound/4, N=1 )

# Control formula components.
C, K, B = distanceBasedControlMatrices( rotz( np.pi/4 )@Aset, m )

# Main execution block.
if __name__ == '__main__':
    # Time series variables.
    T = 1;  Nt = round( T/dt ) + 1
    tList = np.array( [[i*dt for i in range( Nt )]] )

    # Set parameters.
    epsList = [0.0];  Ne = len( epsList )
    vcolor = ['cornflowerblue', 'mediumpurple']
    vlinestyle = Ne*['solid']

    # Initial vehicle positions.
    V0 = np.zeros( (2,1) )

    # Used for plotting without sim.
    yList = np.nan*np.empty( (m,Nt,Nx) )
    xList = np.nan*np.empty( (m,Nt,Nx) )
    VList = np.nan*np.empty( (m,Nt,2) )

    # Initialize simulation variables.
    fig, axs = plt.subplots( 1,Ne+1 )

    # Simulation block.
    X0 = [None for i in range( Ne )]
    Y0 = [None for i in range( Ne )]
    xswrm = [None for i in range( Ne )]
    yswrm = [None for i in range( Ne )]
    cand = [None for i in range( Ne )]
    for i, eps in enumerate( epsList ):
        X0[i] = Abound/2*np.hstack(
            [rotz(k*2*np.pi/m)@[[1],[0]] for k in range( m )]
            ) + noiseCirc( eps=delta, N=m )
        Y0[i] = distanceBasedControl( X0[i], Xeq, C, K, B, A=Aset, eps=eps )[1]

        _, _, xswrm[i], _, _ = initEnvironment(
            fig, [axs[i], axs[-1]], X0[i], Xeq, Aset, V0, Nt=Nt)
        # yswrm[i] = Swarm2D( X0[i], fig=fig, axs=axs[i],
        #     radius=-0.30, color='yellowgreen', tail_length=Nt,
        #     draw_tail=sim ).setLineStyle( '--' ).draw()

    # Simulation block.
    for i, eps in enumerate( epsList ):
        X = X0[i];  Y = Y0[i]
        V0 = lyapunovCandidatePerVehicle( m, 0, X0[i], Xeq )

        xList[:,0,:] = X.T
        yList[:,0,:] = Y.T
        VList[:,0,:] = V0.T
        for t in range( Nt-1 ):
            # Anchor-based control.
            U, Y = distanceBasedControl( X, Xeq, C, K, B, A=Aset, eps=eps )

            # Apply dynamics.
            X = model( X, U )

            # Lyapunov function.
            V = lyapunovCandidatePerVehicle( m, t+1, X, Xeq )

            # Save values.
            xList[:,t+1,:] = X.T
            yList[:,t+1,:] = Y.T
            VList[:,t+1,:] = V.T

        plotEnvironment( fig, [axs[i], axs[-1]], xswrm[i], xList, VList,
            plotXf=False, vcolor=vcolor[i], linestyle=vlinestyle[i], linewidth=1.5 )
        # plotEnvironment( fig, [axs[i], axs[-1]], yswrm[i], yList,
        #     plotXf=False, xcolor='yellowgreen', zorder=z_swrm-100 )

    # Plot and axis labels.
    titles = ['$\\theta=\\pi/4$', 'Lyapunov Trend']
    xlabels = ['$x$', 'Iteration']
    ylabels = ['$y$', '$V(x)$']
    for a, title, xlabel, ylabel in zip( axs, titles, xlabels, ylabels ):
        a.set_title( title )
        a.set_xlabel( xlabel )
        a.set_ylabel( ylabel )

    # Plot transformed grid for reference.
    legend_elements_1 = [
        Line2D([0], [0], color='cornflowerblue', linestyle='none', marker='x',
            label='$X^{(0)}$'),
        Line2D([0], [0], color='cornflowerblue', markerfacecolor='none',
            label='$X$'),
        Line2D([0], [0], color='indianred', linestyle='none', marker='o', markeredgecolor='k',
            label='$\\mathcal{A}$' )
    ]
    axs[1].legend( handles=legend_elements_1, fontsize=fontsize )

    fig.set_figwidth( plt.rcParams.get('figure.figsize')[0] )
    fig.set_figheight( 3/4*figheight )
    fig.tight_layout()
    if show:
        plt.show( block=0 )

    # Calculate error after transformation.
    ans = input( 'Press ENTER to exit program... ' )
    if save or ans == 'save':
        filename = 'single/break.pdf'
        fig.savefig( figurepath + filename, dpi=600 )
        print( 'Figure saved to:\n ' + figurepath + filename )