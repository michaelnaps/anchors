import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *


# Set hyper parameter(s).
Nr = 3                      # Number of anchor sets + reflection sets.
N = 3                       # Number of anchors.
# N = np.random.randint(1,10)
M = Nr*N                    # Number of vehicles.

# Exclusion elements in measurement function.
def exclude(i, j):
    return False # (j - i) % N == 0

# Anchor set.
A = np.array( [[2,5,8],[3,6,9]] )
# A = Abound*np.random.rand( 2,N )
# A = np.array( [
#     [i for i in range( int( Abound + 1 ) ) if i%2 != 0],
#     [i for i in range( int( Abound + 1 ) ) if i%2 != 0]] )
# A = noise( eps=Abound, shape=(2,N) )
print( 'A:\n', A )

# Reflection sets.
Ax = Rx@A
Ay = Ry@A
Q = np.hstack( (A, Ax, Ay) )
print( 'Ax:\n', Ax )
print( 'Ay:\n', Ay )
print( 'Q:\n', Q )

# For error calculation.
Qerr = np.vstack( (Q, np.ones( (1,M) )) )

# Calculate anchor coefficient matrices.
S = signedCoefficientMatrix( N )
Z = anchorCoefficientMatrix( A, N, exclude=exclude )
print( 'S:\n', S )
print( 'Z:\n', Z )


# Main execution block.
if __name__ == '__main__':
    # Simulation time.
    T = 10;  Nt = round( T/dt ) + 1
    tList = np.array( [ [i for i in range( Nt )] ] )

    # Initialize vehicle positions.
    X0 = Q
    epsList = (0, 1, 5, 25, 50)
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
            anchors = Swarm2D( Q, fig=figSim, axs=axsSim, zorder=50,
                radius=R1, draw_tail=False, color='indianred'
                ).setLineStyle( None, body=True ).draw()
            axsSim.axis( 1.5*Abound*np.array( [-1, 1, -1, 1] ) )
            axsSim.axis( 'equal' )
            axsSim.set_title( '$\\varepsilon = %0.1f$' % eps )
            plt.show( block=0 )
            input( "Press ENTER to start sim for eps = %.1f..." % eps )

        # Initial error calculation.
        eTrend[i,0] = formationError( X, Q )[1]

        for j in range( 1,Nt ):
            # Calculate control.
            U = symmetricControl( X, Q, C, Z, S, eps=eps, exclude=exclude )

            # Apply dynamics.
            X = model( X, U )

            if sim:
                swrm.update( X )
                plt.pause( pausesim )

            # Calculate tranformation error.
            eTrend[i,j] = formationError( X, Q )[1]

        # Close simulation plot.
        if sim:
           plt.close( 'all' )

    # Plot error results.
    fig, axs = plt.subplots( 1,2 )
    ymax = np.max( eTrend[-1,:] )
    # fig.suptitle( 'Formation Error' )
    titles = ('Trend', 'Mean')
    xlabels = ('Iteration [n]', 'max$(p(\\varepsilon))$')
    ylabels = ('$|| X - (K Q + k) ||_2$', None)
    for a, t, x, y in zip( axs, titles, xlabels, ylabels ):
        a.grid( 1 )
        a.set_ylim( [0, ymax+0.01] )
        a.set_title( t )
        a.set_xlabel( x )
        a.set_ylabel( y )

    s = round( 1.5/dt )  # Settling time used in mean.
    axs[1].plot( epsList, eTrend[:,s:].mean( axis=1 ), color='grey', marker='.' )
    for eps, error in zip( np.flipud( epsList ), np.flipud( eTrend ) ):
        label = '$\\varepsilon = %0.1f$' % eps
        eAvrg = error[s:].mean()
        axs[0].plot( tList[0], error, linestyle='None', marker='.', markersize=2 )
        axs[1].plot( [0, epsList[-1]], [eAvrg, eAvrg], label=label )

    axs[1].set_xticks( [10*i for i in range( Ne+1 )] )
    handles, labels = axs[1].get_legend_handles_labels()
    axs[1].legend(handles[::-1], labels[::-1])

    fig.tight_layout()
    fig.set_figheight( figheight )
    plt.show( block=0 )

    input( 'Press ENTER to exit the program...' )

    # Exit program.
    if save:
        fig.savefig( figurepath + 'formation_error.png', dpi=1000 )
        print( 'Figure saved.' )
