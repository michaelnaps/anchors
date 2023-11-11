from root import *
from matplotlib.lines import Line2D


# Hyper parameters.
z_axes = 150
z_swrm = 100
z_anch = 50

radius = 0.40

# Plot initializations.
def initVehiclePaths(fig, axs, X, Xeq, Nt=1000, xcolor='cornflowerblue', xeqcolor='indianred'):
    swrm = Swarm2D( X, fig=fig, axs=axs, zorder=z_swrm,
        radius=-radius, color='cornflowerblue', tail_length=Nt,
        draw_tail=sim ).draw()
    axs.plot( Xeq[0], Xeq[1], color='indianred', zorder=int( z_swrm/2 ),
        linestyle='none', marker='x' )
    axs.plot( X[0], X[1], color='cornflowerblue', zorder=z_swrm/2,
        linestyle='none', marker='x' )

    return fig, axs, swrm

def initLyapunovTrend(fig, axs, V, Nt=1000, color='cornflowerblue'):
    swrm = Swarm2D( V, fig=fig, axs=axs,
        radius=0.0, color='cornflowerblue', tail_length=Nt ).draw()
    axs.plot( [0, Nt], [0, 0], color='gray', linestyle='--' )

    return fig, axs, swrm

def initEnvironment(fig, axs, X0, Xeq, Aset, V0, Nt=1000, radius=0.40, anchs=1, connect=0):
    anchors = plotAnchors(fig, axs[0], Aset,
        anchs=anchs, radius=radius, connect=connect)[-1]
    fig, axs[0], swrm = initVehiclePaths( fig, axs[0], X0, Xeq, Nt=Nt )
    fig, axs[1], cand = initLyapunovTrend( fig, axs[1], V0, Nt=Nt)

    # Plot/axis titles.
    bounds = np.vstack( [
        1.5*Abound*np.array( [-1, 1, -1, 1] ),
        np.hstack( [V0[0], Nt, -0.01, V0[1]] ) ] )
    for a, b in zip( axs, bounds ):
        a.axis( b )
        a.grid( 1 )
    axs[0].axis( 'equal' )

    # Show plot.
    fig.tight_layout()
    fig.set_figheight( figheight )

    # Return figure.
    return fig, axs, swrm, anchors, cand

# Environment-related plots.
def plotAxesRotation(fig, axs, R, r, color='grey', shrink=1/3):
    # Axes variables.
    aLength = shrink*Abound
    xaxis = R@([[-aLength, aLength],[0, 0]] + r)
    yaxis = R@([[0, 0],[-aLength, aLength]] + r)

    # Plot axes.
    axs.plot( xaxis[0], xaxis[1], color=color, linestyle='--', zorder=z_axes )
    axs.plot( yaxis[0], yaxis[1], color=color, linestyle='--', zorder=z_axes )
    axs.plot( xaxis[0,1], xaxis[1,1], color=color, marker='o', zorder=z_axes )
    axs.plot( yaxis[0,1], yaxis[1,1], color=color, marker='o', zorder=z_axes )

    # Return plot instances.
    return fig, axs

def plotAnchors(fig, axs, A, radius=0.40, color='indianred', anchs=True, connect=False):
    if not anchs:
        linestyle = '--' if connect else 'none'
        axs.plot( A[0], A[1], linestyle=linestyle, marker='x', color=color )
        return fig, axs, None
    anchors = Swarm2D( A, fig=fig, axs=axs, zorder=z_anch,
        radius=radius, draw_tail=False, color=color
        ).setLineWidth( 1.0 ).setLineStyle( None, body=True ).draw()
    if connect:
        c = Vectors( A, fig=fig, axs=axs, color='gray',
            zorder=int( z_anch/2 ) ).setLineStyle( '--' ).setLineWidth( 1.0 ).draw()
    return fig, axs, anchors

# Lyapunov-related plots.
def plotLyapunovTrend(fig, axs, VList, color='cornflowerblue', linestyle='solid'):
    for V in VList:
        axs.plot( V.T[0], V.T[1], color=color, linestyle=linestyle, linewidth=2.5 )
    axs.axis( np.array( [0.0, np.nanmax( VList[:,:,0] ), 0.0, np.nanmax( VList[:,:,1] )] ) )
    return fig, axs

# Vehicle-related plots.
def plotVehiclePaths(fig, axs, swrm, xList, plotXf=True, color=None, zorder=z_swrm):
    if color is None:
        color = swrm.color

    if not sim:
        if plotXf:
            if len( xList.shape ) == 3:
                swrm.update( xList[:,-1,:].T )
            elif len( xList.shape ) == 2:
                swrm.update( xList[-1] )
        else:
            swrm.remove()
        for X in xList:
            axs.plot( X.T[0], X.T[1], color=color, zorder=zorder )

    return fig, axs, swrm

def plotEnvironment(fig, axs, swrm, xList, VList=None, plotXf=True, zorder=z_swrm, color='cornflowerblue', linestyle='solid'):
    # Plot results of simulation.
    fig, axs[0], swrm = plotVehiclePaths( fig, axs[0], swrm, xList,
        plotXf=plotXf, zorder=zorder )
    if VList is not None:
        fig, axs[1] = plotLyapunovTrend( fig, axs[1], VList,
            color=color, linestyle=linestyle )

    # Return updated figure.
    return fig, axs

def plotConvergenceRate( fig, axs, thList, nList, N, color='indianred' ):
    # Set dimensions.
    Nth = len( thList )

    # Divergence rates.
    for i in range( Nth ):
        theta = thList[i];  brkRatio = nList[i]/N
        axs.plot( [theta, theta], [0, brkRatio], color=color,
            marker='.', markersize=2, linewidth=2.5 )

    # Return figure details.
    return fig, axs