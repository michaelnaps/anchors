import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *


# Set hyper parameter(s).
# N = np.random.randint(1,10)
N = 3                       # Number of anchors.
M = N                       # Number of vehicles.

# Exclusion elements in measurement function.
def exclude(i, j):
    return False # (j - i) % N == 0

# Anchor set.
Aset = np.array( [[-3,5,2],[1,4,-3]] )
# A = Abound*np.random.rand( 2,N )
# A = np.array( [
#     [i for i in range( int( Abound + 1 ) ) if i%2 != 0],
#     [i for i in range( int( Abound + 1 ) ) if i%2 != 0]] )
# A = noise( eps=Abound, shape=(2,N) )
print( 'Aset:\n', Aset )

# For consistency with notes and error calc.
Xeq = Aset
PSI = lambda A: np.vstack( (A, np.ones( (1,A.shape[1]) )) )


# Calculate anchor coefficient matrices.
A, B = anchorDifferenceMatrices(Aset, N=M)
Z, _ = Regressor( A.T@A, np.eye( Nx,Nx ) ).dmd()
K = Z@A.T
print( 'A:', A )
print( 'B:', B )
print( 'K:', K )


# Main execution block.
if __name__ == '__main__':
    # Simulation time.
    T = 10;  Nt = round( T/dt ) + 1
    tList = np.array( [ [i for i in range( Nt )] ] )

    # Initialize vehicle positions.
    X0 = Xeq
    eMax = 10
    epsList = [i for i in range( 0,eMax+1,2 )]
    print( 'eps:\n', epsList )

    # For error trend plotting.
    Ne = len( epsList )
    eTrend = np.empty( (Ne,Nt) )

    # Simulation block.
    for i, eps in enumerate( epsList ):
        # Reset initial conditions.
        X = X0

        # Swarm variable.
        if sim:
            R1 = 0.40
            figSim, axsSim = plt.subplots()
            swrm = Swarm2D( X, fig=figSim, axs=axsSim, zorder=100,
                radius=-R1, color='cornflowerblue', draw_tail=False
                ).draw()
            anchors = Swarm2D( Xeq, fig=figSim, axs=axsSim, zorder=50,
                radius=R1, draw_tail=False, color='indianred'
                ).setLineStyle( None, body=True ).draw()
            axsSim.axis( 1.5*Abound*np.array( [-1, 1, -1, 1] ) )
            axsSim.axis( 'equal' )
            axsSim.set_title( '$\\varepsilon = %0.1f$' % eps )
            plt.show( block=0 )
            input( "Press ENTER to start sim for eps = %.1f..." % eps )

        # Initial error calculation.
        eTrend[i,0] = formationError( X, Xeq )[1]

        for j in range( 1,Nt ):
            # Calculate control.
            U = asymmetricControl( X, Xeq, C, K, B, eps=eps )

            # Apply dynamics.
            X = model( X, U )

            if sim:
                swrm.update( X )
                plt.pause( pausesim )

            # # Calculate tranformation error.
            # eTrend[i,j] = formationError( X, Xeq )[1]
            eTrend[i,j] = np.linalg.norm( U )

        # Close simulation plot.
        if sim:
           plt.close( 'all' )

    # Plot error results.
    fig, axs = plt.subplots( 1,2 )
    ymax = np.max( eTrend[-1] )
    # fig.suptitle( 'Formation Error' )
    titles = ('Trend', 'Mean')
    xlabels = ('Iteration', '$\\varepsilon$')
    # ylabels = ('$|| X - (\\Psi X^{(\\text{eq})} + \\psi) ||_2$', None)
    ylabels = ('$|| U ||_2$', None)
    for a, t, x, y in zip( axs, titles, xlabels, ylabels ):
        a.set_title( t )
        a.set_xlabel( x )
        a.set_ylabel( y )
        a.set_ylim( [0, 1.1*ymax] )
        a.grid( 1 )

    axs[1].plot( epsList, eTrend.mean( axis=1 ), color='grey', marker='.' )
    for eps, error in zip( np.flipud( epsList ), np.flipud( eTrend ) ):
        label = '$\\varepsilon = %0.1f$' % eps
        eAvrg = error.mean()
        axs[0].plot( tList[0], error, linestyle='None', marker='.', markersize=2 )
        axs[1].plot( [0, epsList[-1]], [eAvrg, eAvrg], label=label )

    axs[1].set_xticks( epsList )
    handles, labels = axs[1].get_legend_handles_labels()
    axs[1].legend(handles[::-1], labels[::-1])

    fig.tight_layout()
    fig.set_figheight( figheight )
    plt.show( block=0 )

    input( 'Press ENTER to exit the program...' )

    # Exit program.
    if save:
        fig.savefig( figurepath + 'control_norm_e%i.png' % eMax, dpi=1000 )
        print( 'Figure saved.' )

    # figBox, axsBox = plt.subplots()
    # axsBox.boxplot( eTrend.T )
    # axsBox.set_xticklabels( epsList )
    # plt.show()
