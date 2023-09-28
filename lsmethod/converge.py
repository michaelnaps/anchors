import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *


# Set hyper parameter(s).
# N = np.random.randint(1,10)
N = 6                       # Number of anchors.
M = N                       # Number of vehicles.

# Exclusion elements in measurement function.
def exclude(i, j):
    return False

# Anchor set.
Aset = np.array( [[-1, 1, 1, -4, 4, 4],[1, 1, -1, 4, 4, -4]] )
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
    T = 1;  Nt = round( T/dt ) + 1
    tList = np.array( [ [i for i in range( Nt )] ] )

    # Initialize vehicle positions and error list.
    X0 = Xeq
    eMax = 20
    epsList = np.array( [i for i in range( eMax+1 )] )
    infCount = {eps: 0 for eps in epsList}
    print( 'eps:\n', epsList )

    # For error trend plotting.
    Ne = len( epsList )
    Ni = 100
    eTrend = np.empty( (Ne*Ni,Nt) )

    # Simulation block.
    k = 0
    for i, eps in enumerate( epsList ):
        # Repitition block.
        for _ in range( Ni ):
            # Reset initial conditions.
            X = X0 + noiseCirc( eps=eps, N=M )

            # Initial error calculation.
            eTrend[k,0] = formationError( X, Xeq )[1]

            # SModel block..
            for j in range( 1,Nt ):
                # Calculate control.
                U = asymmetricControl( X, Xeq, C, K, B )

                # Apply dynamics.
                X = model( X, U )

                # Calculate error and break if too large.
                # eTrend[k,j] = np.linalg.norm( U )
                eTrend[k,j] = formationError( X, Xeq )[1]
                if eTrend[k,j] > 10*eps and eps != 0:
                    eTrend[k,j:] = np.inf
                    infCount[eps] += 1
                    break

            # Update iterator.
            k += 1

    # Maximum axis value.
    eTotal = eTrend.reshape(Nt*Ne*Ni)
    ymax = np.max( eTotal[np.isfinite(eTotal)] )

    # Plot error results.
    fig, axs = plt.subplots( 1,2 )
    # fig.suptitle( 'Formation Error' )
    titles = (None, None)
    xlabels = ('Iteration', '$\\varepsilon$')
    ylabels = ('$|| X - (\\Psi X^{(\\text{eq})} + \\psi) ||_2$', '\% Diverged')
    # ylabels = ('$|| U ||_2$', None)
    for a, t, x, y in zip( axs, titles, xlabels, ylabels ):
        a.set_title( t )
        a.set_xlabel( x )
        a.set_ylabel( y )
        a.grid( 1 )

    # axs[1].plot( epsList, eTrend.mean( axis=1 ), color='grey', marker='.' )
    for eps, error in zip( epsList[::-1], eTrend[::-1] ):
        label = '$\\varepsilon = %0.1f$' % eps
        brkRatio = infCount[eps]/Ni
        axs[0].plot( tList[0], error, marker='.', markersize=2 )
        axs[1].plot( [eps, eps], [0, brkRatio], linewidth=3, label=label )

    # Legend and axis ticks.
    axs[1].set_xticks( epsList[::2] )
    handles, labels = axs[1].get_legend_handles_labels()
    # axs[1].legend(handles[::-1], labels[::-1])

    fig.tight_layout()
    fig.set_figheight( figheight )
    plt.show( block=0 )

    # Exit program.
    ans = input( 'Press ENTER to exit the program... ' )
    if save or ans == 'save':
        fig.savefig( figurepath + 'convergence_e%i.png' % eMax, dpi=1000 )
        print( 'Figure saved.' )

    # figBox, axsBox = plt.subplots()
    # axsBox.boxplot( eTrend.T )
    # axsBox.set_xticklabels( epsList )
    # plt.show()
