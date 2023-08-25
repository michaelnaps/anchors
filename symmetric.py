
from root import *


# Controller gains of interest.
C = 10*np.diag( np.random.rand( Nx, ) )


# Anchor values.
Na = 3
q = 2*A*np.random.rand( 2,1 ) - A

print( 'number of anchors: ', Na )
print( 'desired position: ', q.T )


# Anchor and reflection sets.
S = np.array( [[5, -5, 5],[5, 5, -5]] )
aList = S[:,0,None]
rxList = S[:,1,None]
ryList = S[:,2,None]


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
def anchorControl(x):
    # Combine anchor sets.
    dList = anchorMeasure( x )
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
    tswrm = Swarm2D( X0, fig=fig, axs=axs, radius=R, color='yellowgreen', tail_length=100 ).draw()
    aswrm = Swarm2D( X0, fig=fig, axs=axs, radius=R/2, color='indianred', tail_length=100 )
    aswrm.setLineStyle( '--' ).draw()

    # Anchor plotting.
    anchors = Swarm2D( aList, fig=fig, axs=axs, radius=0.25, color='yellowgreen', draw_tail=0 ).draw()
    xreflect = Swarm2D( rxList, fig=fig, axs=axs, radius=0.25, color='orange', draw_tail=0 ).draw()
    yreflect = Swarm2D( ryList, fig=fig, axs=axs, radius=0.25, color='indianred', draw_tail=0 ).draw()

    # Axis setup.
    plt.axis( [-10, 10, -10, 10] )
    plt.gca().set_aspect( 'equal', adjustable='box' )
    plt.show( block=0 )

    # Simulation parameters.
    T = 10;  Nt = round( T/dt ) + 1
    tList = np.array( [ [i*dt for i in range( Nt )] ] )

    # Simulation loop.
    xtrue = X0
    xanch = X0
    uanch = np.empty( (Nu,N0) )
    for t in tList.T:
        utrue = control( xtrue, C=C, q=q )
        for i, x in enumerate( xanch.T ):
            uanch[:,i] = anchorControl( x[:,None] )[:,0]

        xtrue = model( xtrue, utrue )
        xanch = model( xanch, uanch )

        tswrm.update( xtrue )
        aswrm.update( xanch )
        plt.pause( sim_pause )

        if np.linalg.norm( utrue + uanch ) < 0.1:
            print( 'No motion, ending simulation early.' )
            break

    input( 'Press ENTER to exit program. ' )
