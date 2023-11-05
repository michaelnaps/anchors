from root import *
from matplotlib.lines import Line2D

# Plotting-related members.
def plotAnchors(fig, axs, A, radius=0.40, connect=False, color='indianred'):
    anchors = Swarm2D( A, fig=fig, axs=axs, zorder=50,
        radius=radius, draw_tail=False, color=color
        ).setLineWidth( 1.0 ).setLineStyle( None, body=True ).draw()
    if connect:
        c = Vectors( A, fig=fig, axs=axs,
            zorder=1, color='gray'
        ).setLineStyle( '--' ).setLineWidth( 1.0 ).draw()
    return anchors

def initAnchorEnvironment(X, Xeq, A, V0, Nt=1000, ge=1, radius=0.40, delta=0.00, anchs=True, dist=True, fig=None, axsX=None, axsV=None):
    # Plot initialization.
    if fig is None:
        fig, axs = plt.subplots( 1,2 )
        axsX = axs[0];  axsV = axs[1]
    else:
        axs = (axsX, axsV)

    # Optionally plot anchors.
    if anchs:
        anchors = plotAnchors(fig, axsX, A, radius)
    else:
        anchors = None

    if dist:
        disturb = Swarm2D( Xeq, fig=fig, axs=axsX, zorder=10,
            radius=delta, draw_tail=False, color='none'
            ).setLineStyle( ':', body=True
            ).setLineWidth( 1.0, body=True ).draw()
    else:
        disturb = None

    # Swarm variables.
    swrm = Swarm2D( X, fig=fig, axs=axsX, zorder=100,
        radius=-radius, color='cornflowerblue', tail_length=Nt,
        draw_tail=sim ).draw()
    axsX.plot( Xeq[0], Xeq[1], zorder=50, color='indianred',
        linestyle='none', marker='x' )
    axsX.plot( X[0], X[1], zorder=50, color='cornflowerblue',
        linestyle='none', marker='x' )

    # For plotting error.
    error = Vehicle2D( V0, fig=fig, axs=axsV,
        radius=0.0, color='cornflowerblue', tail_length=Nt ).draw()
    axsV.plot( [0, ge*Nt], [0, 0], color='gray', linestyle='--' )

    # Axis setup.
    titles = ('Environment', 'Lyapunov Trend')
    xlabels = ('$\\mathbf{x}$', 'Iteration')
    # ylabels = ('$\\mathbf{y}$', '$|| X - (\\Psi X^{(\\text{eq})} + \\psi) ||_2$')
    ylabels = ('$\\mathbf{y}$', '$V(\\Psi X + \\psi)$')
    bounds = np.vstack( [
        1.5*Abound*np.array( [-1, 1, -1, 1] ),
        np.hstack( [V0[0], ge*Nt, -0.01, V0[1]] ) ] )
    for i in range( 2 ):
        axs[i].set_title( titles[i] )
        axs[i].set_xlabel( xlabels[i] )
        axs[i].set_ylabel( ylabels[i] )
        axs[i].axis( bounds[i] )
        axs[i].grid( 1 )
    axs[0].axis( 'equal' )

    # Legend formation.
    legend_elements = [
        Line2D([0], [0], color='cornflowerblue', linestyle='none', marker='x',
            label='$X^{(0)}$'),
        Line2D([0], [0], color='cornflowerblue', marker='o', markerfacecolor='none',
            label='$X$'),
        Line2D([0], [0], color='indianred', linestyle='none', marker='x',
            label='$\\mathcal{A}$, $X^{(\\text{eq})}$' ),
        Line2D([0], [0], color='grey', linestyle='--',
            label='$\\Psi [\\mathbf{x}, \\mathbf{y}]^\\top + \\psi$' ),
    ]
    axsV.legend( handles=legend_elements, ncol=1 )

    # Show plot.
    fig.tight_layout()
    fig.set_figheight( figheight )
    plt.show( block=0 )

    # Return figure.
    return fig, axs, swrm, anchors, error

def finalAnchorEnvironment( fig, axs, swrm, xList, VList, Psi, Xbar, shrink=1/3 ):
    if not sim:
        swrm.update( xList[:,-1,:].T )
        axs[1].plot( VList[0], VList[1], color='cornflowerblue' )
    for vhc in xList:
        axs[0].plot( vhc.T[0], vhc.T[1], color='cornflowerblue' )

    xaxis = shrink*Psi@( [[-Abound, Abound],[0, 0]] + Xbar )
    yaxis = shrink*Psi@( [[0, 0],[-Abound, Abound]] + Xbar )

    axs[0].plot( xaxis[0], xaxis[1], color='grey', linestyle='--', zorder=150 )
    axs[0].plot( yaxis[0], yaxis[1], color='grey', linestyle='--', zorder=150 )
    axs[0].plot( xaxis[0,1], xaxis[1,1], color='grey', marker='o', zorder=150 )
    axs[0].plot( yaxis[0,1], yaxis[1,1], color='grey', marker='o', zorder=150 )
    axs[1].axis( np.array( [0.0, max( VList[0] ), 0.0, max( VList[1] )] ) )

    # Return figure.
    return fig, axs

def finalAnchorEnvironmentAnchored( fig, axs, xswrm, yswrm, xList, yList, VList, shrink=1/3 ):
    if not sim:
        print( xList )
        xswrm.update( xList[:,-1,:].T )
        if yswrm is not None:
            yswrm.update( yList[:,-1,:].T )
        axs[1].plot( VList[0], VList[1], color='cornflowerblue' )
    for xvhc in xList:
        axs[0].plot( xvhc.T[0], xvhc.T[1], color='cornflowerblue', zorder=50 )
    if yList is not None:
        for yvhc in yList:
            axs[0].plot( yvhc.T[0], yvhc.T[1], color='yellowgreen',
                linewidth=1, zorder=10 )
    axs[1].axis( np.array( [0.0, max( VList[0] ), 0.0, max( VList[1] )] ) )

    # Return figure.
    return fig, axs
