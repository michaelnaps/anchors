import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *

A = Abound

# Anchor values.
# Na = 1
# Na = 100
Na = np.random.randint( 1,15 )

# Desired position term.
# q = 7.5*np.ones( (Nx,1) )
# q = 2*A*np.random.rand( 2,1 ) - A
q = np.array( [ [1],[-5] ])
print( 'number of anchors: ', 3*Na )
print( 'desired position: ', q.T )

# Anchor and reflection sets.
aList = noise( eps=A, shape=(2,Na) )
rxList = Rx@aList
ryList = Ry@aList

# Control matrices.
D = -1/4*np.diag( [ 1/np.sum( aList[0] ), 1/np.sum( aList[1] ) ] )
print( 'Anchor coefficient matrix:\n', D )


# Anchor measurement functions.
def anchorMeasure(x):
    d = np.empty( (1,Na) )
    for i, a in enumerate( aList.T ):
        d[:,i] = (x - a[:,None]).T@(x - a[:,None])
    return np.sqrt( d )

def reflectionMeasure(x):
    dr = np.empty( (2,Na) )
    for i, (rx, ry) in enumerate( zip( rxList.T, ryList.T ) ):
        dr[:,i] = np.array( [
            (x - rx[:,None]).T@(x - rx[:,None]),
            (x - ry[:,None]).T@(x - ry[:,None]),
        ] )[:,0,0]
    return np.sqrt( dr )


# Anchor-based control policy.
def anchorControl(x, eps=0):
    # Combine anchor sets.
    dList = anchorMeasure( x ) + noise( eps=eps, shape=(1,Na) )
    drList = reflectionMeasure( x ) + noise( eps=eps, shape=(2,Na) )

    # Calculate measurement state.
    z = np.array( [
        np.sum( dList**2 - drList[1]**2, axis=1 ),
        np.sum( dList**2 - drList[0]**2, axis=1 )
    ] )

    # Return control.
    return C@(q - D@z)


# Main execution block.
if __name__ == '__main__':
    # Simulation parameters.
    T = 10;  Nt = round( T/dt ) + 1
    tList = np.array( [ [i*dt for i in range( Nt )] ] )

    # Initial state terms.
    N0 = 10;  B = A
    X0 = 2*B*np.random.rand( Nx,N0 ) - B

    # Initial position.
    xtrue = X0
    xanch = X0 + noise( eps=1.0, shape=(Nx,N0) )

    # Example simulation.
    fig, axs = plt.subplots()
    axs.plot( q[0], q[1], color='g', marker='x' )
    R = 0.50
    tswrm = Swarm2D( xtrue, fig=fig, axs=axs, zorder=125,
        radius=R, color='yellowgreen', tail_length=Nt ).draw()
    aswrm = Swarm2D( xanch, fig=fig, axs=axs, zorder=150,
        radius=R/2, color='yellowgreen', draw_tail=False ).draw()
    # aswrm.setLineStyle( '--' ).draw()

    # Anchor plotting.
    anchors = Swarm2D( aList, fig=fig, axs=axs, zorder=1,
        radius=0.25, color='indianred', draw_tail=0 ).draw()
    xreflect = Swarm2D( rxList, fig=fig, axs=axs, zorder=1,
        radius=0.25, color='cornflowerblue', draw_tail=0 ).draw()
    yreflect = Swarm2D( ryList, fig=fig, axs=axs, zorder=1,
        radius=0.25, color='mediumpurple', draw_tail=0 ).draw()

    # Axis setup.
    plt.axis( 10*np.array( [-1, 1, -1, 1] ) )
    plt.gca().set_aspect( 'equal', adjustable='box' )
    plt.xticks( [-i for i in range( -A,A+1 )] )
    plt.yticks( [-i for i in range( -A,A+1 )] )
    plt.show( block=0 )

    # Simulation loop.
    uanch = np.empty( (Nu,N0) )
    for t in tList.T:
        # Calculate control for each vehicle.
        for i, x in enumerate( xtrue.T ):
            uanch[:,i] = anchorControl( x[:,None], eps=1.0 )[:,0]

        xtrue = model( xtrue, uanch )
        # xanch = model( xanch, uanch )

        tswrm.update( xtrue )
        # aswrm.update( xanch )
        plt.pause( sim_pause )

        if np.linalg.norm( uanch ) < 0.1:
            print( 'No motion, ending simulation early.' )
            break

    input( 'Press ENTER to exit program. ' )
