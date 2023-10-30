import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *


# Set hyper parameter(s).
# N = np.random.randint(1,10)
N = 3                       # Number of anchors.
M = N                       # Number of vehicles.

# Plot color ranges.
def getColorError(bk):
    if bk:
        return 'indianred'
    return 'cornflowerblue'

def getColorTheta(th):
    # cList = ['yellowgreen', 'indianred', 'cornflowerblue']
    # iColor = round( abs(th)/np.pi )
    return 'indianred'

# Anchor set.
Aset = Abound*np.array( [[-1, 1, 1],[1, 1, -1]] )
# Aset = noiseCirc( eps=Abound, N=N )
print( 'Aset:\n', Aset )

# For consistency with notes and error calc.
Xeq = Aset

# Control formula components.
C, K, B = distanceBasedControlMatrices( Aset, M )

# Main execution block.
if __name__ == '__main__':
    # Simulation time.
    T = 5.0;  Nt = round( T/dt ) + 1
    tList = np.array( [ [i for i in range( Nt )] ] )

    # Initialize vehicle positions and error list.
    Nth = 25
    thList = np.linspace( -2*np.pi,2*np.pi,Nth )
    rotList = [rotz(theta) for theta in thList]
    infCount = {i: [thList[i], 0] for i in range( Nth )}
    # print( 'theta:\n', thList )
    # print( 'infCount:\n', infCount )

    # For error trend plotting.
    Ni = 10
    delta = 10
    eTrend = np.empty( (Nth*Ni,Nt) )

    # Simulation block.
    k = 0
    for i, R in enumerate( rotList ):
        # Repitition block.
        for _ in range( Ni ):
            # Reset initial conditions.
            X = R@Xeq + noiseCirc( eps=delta, N=M )

            # Initial error calculation.
            eTrend[k,0] = lyapunovCandidate( X, Xeq )

            # SModel block..
            for j in range( 1,Nt ):
                # Calculate control.
                U = distanceBasedControl( X, Xeq, C, K, B )[0]

                # Apply dynamics.
                X = model( X, U )

                # Calculate error and break if too large.
                # eTrend[k,j] = np.linalg.norm( U )
                eTrend[k,j] = lyapunovCandidate( X, Xeq )
                if (eTrend[k,j] > 20 and delta != 0) :
                    eTrend[k,j:] = np.inf
                    infCount[i][1] += 1
                    break
                elif np.linalg.norm( U ) < 1e-3:
                    eTrend[k,j:] = eTrend[k,j]

            # Update iterator.
            k += 1

    # Maximum axis value.
    eTotal = eTrend.reshape(Nt*Nth*Ni)
    ymax = np.max( eTotal[np.isfinite(eTotal)] )

    # Plot error results.
    fig, axs = plt.subplots()
    # fig.suptitle( 'Formation Error' )
    titles = (None, None)
    xlabels = ('Iteration', '$\\theta$')
    ylabels = ('$|| X - (\\Psi X^{(\\text{eq})} + \\psi) ||_2$', '\% Diverged')
    # for a, t, x, y in zip( axs, titles, xlabels, ylabels ):
    #     a.set_title( t )
    #     a.set_xlabel( x )
    #     a.set_ylabel( y )
    #     a.grid( 1 )

    axs.set_title( titles[1] )
    axs.set_xlabel( xlabels[1] )
    axs.set_ylabel( ylabels[1] )
    axs.grid( 1 )

    # for i, error in enumerate( eTrend ):
    #     axs[0].plot( tList[0], error,
    #         color=getColorError( not np.isfinite( error[-1] ) ),
    #         marker='.', markersize=2 )

    for key in infCount.keys():
        label = '$\\theta = %0.1f$' % infCount[key][0]
        theta = infCount[key][0]
        brkRatio = infCount[key][1]/Ni
        axs.plot( [theta, theta], [0, brkRatio],
            color=getColorTheta( theta ),
            marker='.', markersize=2, linewidth=2.5, label=label )

    bound_nums = np.array( [k*np.pi/2 for k in range(-3,4,2)] )
    bound_labels = [
        '$\\frac{-3 \\pi}{2}$',
        '$\\frac{-\\pi}{2}$',
        '$\\frac{\\pi}{2}$',
        '$\\frac{3 \\pi}{2}$' ]
    for bound in bound_nums:
        axs.plot([bound, bound], [0, 1], color='k', linestyle='--')

    # Legend and axis ticks.
    axs.set_xticks(bound_nums, bound_labels)
    handles, labels = axs.get_legend_handles_labels()

    fig.tight_layout()
    fig.set_figheight( figheight )
    plt.show( block=0 )

    # Exit program.
    ans = input( 'Press ENTER to exit the program... ' )
    if save or ans == 'save':
        fig.savefig( figurepath + f'multi/bounds_n{N}_d{delta}_r{Ni}.png', dpi=1000 )
        print( 'Figure saved.' )
