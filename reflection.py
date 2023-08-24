
from anchors import *


# Controller gains of interest.
C = 10*np.diag( np.random.rand( Nx, ) )


# Anchor values.
Na = np.random.randint(2, 100)
# Na = 3
q = 2*A*np.random.rand( 2,1 ) - A

print( 'number of anchors: ', Na )
print( 'desired position: ', q.T )


# Anchor and reflection sets.
aList = np.random.rand( 2,Na )
rxList = np.vstack( (-aList[0], aList[1]) )
ryList = np.vstack( (aList[0], -aList[1]) )


# Control matrices.
D = 1/4*np.diag( [ 1/np.sum( aList[0] ), 1/np.sum( aList[1] ) ] )


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
    return C@D@d + C@q


# Main execution block.
if __name__ == '__main__':
    # Initial state terms.
    N0 = 10;  B = 10
    X0 = 2*B*np.random.rand( Nx,N0 ) - B

    # Example simulation.
    fig, axs = plt.subplots()
    axs.plot( q[0], q[1], color='g', marker='x' )
    R = 0.50
    tswrm = Swarm2D( X0, fig=fig, axs=axs, radius=R, color='k', tail_length=100 )
    aswrm = Swarm2D( X0, fig=fig, axs=axs, radius=0.75*R, color='indianred', tail_length=100 )
    aswrm.setLineStyle( '--' )
    tswrm.draw()
    aswrm.draw()

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
        plt.pause( 1e-3 )

        if np.linalg.norm( utrue + uanch ) < 1e-6:
            print( 'No motion, ending simulation early.' )
            break
