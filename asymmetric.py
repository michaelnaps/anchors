
from root import *


# Controller gains of interest.
C = np.eye( Nx )
# C = 10*np.diag( np.random.rand( Nx, ) )

# Anchor values.
Na = 25
# Na = np.random.randint(1, 15)

# Desired position term.
q = 7.5*np.ones( (Nx,1) )
# q = 2*A*np.random.rand( 2,1 ) - A
print( 'number of anchors: ', Na )
print( 'desired position: ', q.T )

# Anchor and reflection sets.
aList = 2*A*np.random.rand( 2,Na ) - A
rxList = np.vstack( (-aList[0], aList[1]) )
ryList = np.vstack( (aList[0], -aList[1]) )

# Control matrices.
D = 1/4*np.diag( [ 1/np.sum( aList[0] ), 1/np.sum( aList[1] ) ] )
print( 'Anchor coefficient matrix:\n', C@D )


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
    drList = reflectionMeasure( x )

    # Calculate measurement state.
    d = np.array( [
        np.sum( dList**2 - drList[0]**2, axis=1 ),
        np.sum( dList**2 - drList[1]**2, axis=1 )
    ] )

    # Return control.
    return C@(D@d + q)


# Main execution block.
if __name__ == '__main__':
    # Initial state terms.
    N0 = 10;  B = 10
    X0 = 2*B*np.random.rand( Nx,N0 ) - B

    # Example simulation.
    fig, axs = plt.subplots()
    axs.plot( q[0], q[1], color='g', marker='x' )
    R = 0.50
    tswrm = Swarm2D( X0, fig=fig, axs=axs, zorder=100,
        radius=R, color='yellowgreen', tail_length=100 ).draw()
    aswrm = Swarm2D( X0, fig=fig, axs=axs, zorder=10,
        radius=R/2, color='indianred', tail_length=100 )
    aswrm.setLineStyle( '--' ).draw()

    # Anchor plotting.
    anchors = Swarm2D( aList, fig=fig, axs=axs,
        radius=0.25, color='yellowgreen', draw_tail=0 ).draw()
    xreflect = Swarm2D( rxList, fig=fig, axs=axs,
        radius=0.25, color='grey', draw_tail=0 ).draw()
    yreflect = Swarm2D( ryList, fig=fig, axs=axs,
        radius=0.25, color='grey', draw_tail=0 ).draw()

    # Axis setup.
    plt.axis( [-10, 10, -10, 10] )
    plt.gca().set_aspect( 'equal', adjustable='box' )
    plt.show( block=0 )

    # Simulation parameters.
    T = 10;  Nt = round( T/dt ) + 1
    tList = np.array( [ [i*dt for i in range( Nt )] ] )

    # Simulation loop.
    xtrue = X0
    xanch = X0 + noise( eps=1, shape=(Nx,1) )
    uanch = np.empty( (Nu,N0) )
    for t in tList.T:
        # Calculate control for each vehicle.
        for i, x in enumerate( xtrue.T ):
            uanch[:,i] = anchorControl( x[:,None], eps=2.5 )[:,0]

        xtrue = model( xtrue, uanch )
        xanch = model( xanch, uanch )

        tswrm.update( xtrue )
        aswrm.update( xanch )
        plt.pause( sim_pause )

        if np.linalg.norm( uanch ) < 0.1:
            print( 'No motion, ending simulation early.' )
            break

    input( 'Press ENTER to exit program. ' )
