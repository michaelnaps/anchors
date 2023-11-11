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
thList = np.linspace( -2*np.pi,2*np.pi,Nth )
RList = np.array( [rotz( theta ) for theta in thList] )

# eps = 0.1
# Nth = 100
# thList = np.linspace( np.pi/2-eps,np.pi/2+eps,Nth )

# Main execution block.
if __name__ == '__main__':
    # Time series variables.
    T = np.inf

    # Break count list.
    Ni = 100
    nList = np.zeros( (Nth+1,) )

    # Rotation loop.
    delta = 0.001
    for i, R in enumerate( RList ):
        # Repitition loop.
        for j in range( Ni ):
            # Simulation loop.
            t = 0;  con = 1  # Default: assume policy converged.
            x = circ( delta, t=noise( eps=2*np.pi, shape=(1,1) ) )
            while t < T:
                # Anchor-based control.
                u = distanceBasedControl( x, xeq, C, K, B, A=R@Aset )[0]

                # Apply dynamics.
                x = model( x, u )

                # Lyapunov function.
                V = lyapunovCandidateAnchored( x, xeq, R=R )

                # Check for convergence/divergence of Lyapunov candidate.
                t += 1
                if V**2 > 1e3:
                    con = 0
                    nList[i] = nList[i] + 1
                    break
                elif V**2 < 1e-24:
                    break

            print( 'Iteration stopped:', (con, t) )

    # Plot results of break simulation.
    fig, axs = plt.subplots()
    fig.set_figheight( 3/4*figheight )
    axs.grid( 1 )

    # b = 0.01
    # axs.axis( [thList[0]-b, thList[-1]+b, 0, 1+b] )
    axs.set_xlabel( '$\\theta$' )
    axs.set_ylabel( 'Ratio Divergence' )
    fig.tight_layout()

    # Divergence rates.
    for i in range( Nth ):
        theta = thList[i];  brkRatio = nList[i]/Ni
        axs.plot( [theta, theta], [0, brkRatio], color='indianred',
            marker='.', markersize=2, linewidth=2.5 )

    # x-label ticks and boundary bars.
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

    plt.show( block=0 )
    ans = input( 'Press ENTER to exit program... ' )
    if save or ans == 'save':
        filename = 'single/rotation.png'
        fig.savefig( figurepath + filename, dpi=600 )
        print( 'Figure saved to:\n ' + figurepath + filename )