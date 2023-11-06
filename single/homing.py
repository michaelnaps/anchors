import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from plotfuncs import *


# Anchor initialization.
n = 3
m = 20
Aset = Abound/2*np.array( [
    [-1, 1, 1],
    [1, 1, -1] ] )

# For consistency with notes and error calc.
Xeq = np.array( [[0],[0]] )  # noiseCirc( eps=Abound/4, N=1 )

# Control formula components.
C, K, B = distanceBasedControlMatrices( Aset, m )

# Main execution block.
if __name__ == '__main__':
    # Time series variables.
    T = 1;  Nt = round( T/dt ) + 1
    tList = np.array( [[i*dt for i in range( Nt )]] )

    # Initial vehicle positions.
    delta = 0.0
    epsList = [0.0, 5.0, 10.0]
    Ne = len( epsList )
    # X0 = 3*Abound/4*np.hstack(
    #     [rotz(k*2*np.pi/m)@[[1],[0]] for k in range( m )]
    #     ) + noiseCirc( eps=delta, N=m )
    X0 = np.hstack( (
        [rotz( 2*np.pi*k/m - np.pi/2 )@[[k/2],[0]] for k in range( 1,m+1 )] ) )
    V0 = np.zeros( (2,1) )

    # Used for plotting without sim.
    xList = np.nan*np.ones( (m,Nt,Nx) )
    yList = np.nan*np.ones( (m,Nt,Nx) )
    VList = np.nan*np.ones( (m,Nt,2) )

    # Initialize simulation variables.
    fig, axs = plt.subplots( 1,Ne+1 )
    fig.set_figwidth( 3*plt.rcParams.get('figure.figsize')[0] )
    vcolor = ['cornflowerblue', 'mediumseagreen', 'mediumpurple']
    vlinestyle = ['solid', '--', ':']

    xswrm = [None for i in range( Ne )]
    yswrm = [None for i in range( Ne )]
    cand = [None for i in range( Ne )]
    for i, eps in enumerate( epsList ):
        _, _, xswrm[i], _, _ = initEnvironment(
            fig, [axs[i], axs[-1]], X0, Xeq, Aset, V0, Nt=Nt)
        yswrm[i] = Swarm2D( X0, fig=fig, axs=axs[i], zorder=z_swrm-100,
            radius=-0.30, color='yellowgreen', tail_length=Nt,
            draw_tail=sim ).setLineStyle( '--' ).draw()

    # Simulation block.
    for i, eps in enumerate( epsList ):
        X = X0;  Y = X0
        V0 = lyapunovCandidatePerVehicle( m, 0, X0, Xeq )

        xList[:,0,:] = X0.T
        yList[:,0,:] = X0.T
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
            plotXf=False, color=vcolor[i], linestyle=vlinestyle[i] )
        plotEnvironment( fig, [axs[i], axs[-1]], yswrm[i], yList,
            plotXf=False, zorder=z_swrm-100 )

    # Axis and plot labels.
    # titles = ('Environment', 'Lyapunov Trend')
    # xlabels = ('$\\mathbf{x}$', 'Iteration')
    # ylabels = ('$\\mathbf{y}$', '$V(\\Psi X + \\psi)$')
    # axs[i].set_title( titles[i] )
    # axs[i].set_xlabel( xlabels[i] )
    # axs[i].set_ylabel( ylabels[i] )

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
    axs[0].axis( Abound*np.array( [-1, 1, -1.15, 1.1] ) )
    axs[0].legend( handles=legend_elements_1, ncol=2, loc=1 )

    legend_elements_2 = [
        Line2D([0], [0], color=vcolor[i], linestyle=vlinestyle[i], linewidth=2,
            label='$\\varepsilon = %.1f$' % eps) for i, eps in enumerate( epsList )
    ]
    axs[-1].legend( handles=legend_elements_2, ncol=1 )

    fig.tight_layout()
    plt.pause( pausesim )

    # Calculate error after transformation.
    print( '\nError: ', VList[1,np.isfinite(VList[1])][-1] )
    ans = input( 'Press ENTER to exit program... ' )
    if save or ans == 'save':
        filename = 'single/' + '_'.join( ['homing'] + ['e%i' % eps for eps in epsList] ) + '.png'
        fig.savefig( figurepath + filename, dpi=600 )
        print( 'Figure saved to:\n ' + figurepath + filename )