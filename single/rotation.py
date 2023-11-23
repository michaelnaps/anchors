import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from plotfuncs import *


# Anchor initialization.
n = 4
m = 1
Aset = Abound/4*np.array( [
    [-1, 1, 1, -1],
    [1, 1, -1, -1] ] )

# For consistency with notes and error calc.
xeq = np.array( [[0],[0]] )  # noiseCirc( eps=Abound/4, N=1 )

# Control formula components.
C, K, B = distanceBasedControlMatrices( Aset, m )

# Rotation list.
Nth = 100
thList = np.linspace( -2*np.pi, 2*np.pi, Nth )
RList = np.array( [rotz( theta ) for theta in thList] )

# Main execution block.
if __name__ == '__main__':
    # Time series variables.
    T = np.inf

    # Break count list.
    Ni = 1
    nList = np.zeros( (Nth,) )

    # Rotation loop.
    delta = 0.001
    for i, R in enumerate( RList ):
        # Repitition loop.
        for j in range( Ni ):
            # Simulation loop.
            t = 0;  con = 1  # Default: assume policy converged.
            x = circ( delta, noise( eps=2*np.pi, shape=(1,1) ) )
            while t < T:
                # Anchor-based control.
                u = distanceBasedControl( x, xeq, C, K, B, A=R@Aset )[0]

                # Apply dynamics.
                x = model( x, u )

                # Lyapunov function.
                V = lyapunovCandidateAnchored( x, xeq, R=R )

                # Check for convergence/divergence of Lyapunov candidate.
                t += 1
                if V > 1e3:
                    con = 0
                    nList[i] = nList[i] + 1
                    break
                elif V < 1e-24:
                    break

            print( 'Iteration stopped:', (con, t) )

    # Plot results of break simulation.
    fig, axs = plt.subplots()
    fig, axs = plotConvergenceRate( fig, axs, thList, nList, Ni )

    # Axes and plot labels.
    axs.set_xlabel( '$\\theta$' )
    axs.set_ylabel( '$\\frac{\\text{\# diverged}}{\\text{\# tests run}}$' )
    bound_nums = np.array( [-3*np.pi/2, -np.pi/2, np.pi/2, 3*np.pi/2] )
    bound_labels = [
        '$-\\frac{3 \\pi}{2}$',
        '$-\\frac{\\pi}{2}$',
        '$\\frac{\\pi}{2}$',
        '$\\frac{3 \\pi}{2}$' ]
    for bound in bound_nums:
        axs.plot( [bound, bound], [0, 1], color='gray', linestyle='--' )
    axs.plot( [thList[0], thList[-1]], [1, 1], color='k', linestyle='--' )
    axs.set_xticks( bound_nums, bound_labels )
    handles, labels = axs.get_legend_handles_labels()
    axs.grid( 1 )

    fig.set_figwidth( 1.0*plt.rcParams.get('figure.figsize')[0] )
    fig.set_figheight( 3/4*figheight )
    fig.tight_layout()
    if show:
        plt.show( block=0 )

    ans = input( 'Press ENTER to exit program... ' )
    if save or ans == 'save':
        filename = 'single/rotation.pdf'
        fig.savefig( figurepath + filename, dpi=600 )
        print( 'Figure saved to:\n ' + figurepath + filename )