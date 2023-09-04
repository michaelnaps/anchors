import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *


# Anchor values.
N = 2    # Number of anchors.
M = 3*N  # Number of vehicles.

# Anchor and corresponding reflection sets.
A = Abound*np.random.rand( 2,N )
Ax = Rx@A
Ay = Ry@A
print( 'A:\n', A )
print( 'Ax:\n', Ax )
print( 'Ay:\n', Ay )

# # Anchor-distance control matrices.
# D = -1/4*np.diag( [ 1/np.sum( A[0] ), 1/np.sum( A[1] ) ] )
# print( 'Anchor coefficient matrices:\n', D )


# Anchor measurement functions.
def anchorMeasure(x, aList=None):
    if aList is None:
        aList = A
    Na = aList.shape[1]
    d = np.empty( (1,Na) )
    for i, a in enumerate( aList.T ):
        d[:,i] = (x - a[:,None]).T@(x - a[:,None])
    return np.sqrt( d )


# Main execution block.
if __name__ == '__main__':
    # Initialize vehicle positions.
    X = np.hstack( (
        A + noise( eps=1e-3, shape=(2,N) ),
        Ax + noise( eps=0.1, shape=(2,N) ),
        Ay + noise( eps=0.1, shape=(2,N) ),
    ) )
    print( 'X:\n', X.T )

    # Initialize control sets.
    k = 0
    Z = np.empty( (M,2,2) )
    for i, x in enumerate( X.T ):
        if k == N:  k = 0
        Atemp = np.array( [A[:,j] for j in range( N ) if k != j] ).T
        Z[i] = -1/4*np.diag( [ 1/np.sum( Atemp[0] ), 1/np.sum( Atemp[1] ) ] )
        k += 1
    print( 'Z:\n', Z )

    # Swarm variables.
    R = 0.25
    fig, axs = plt.subplots()
    swrm = Swarm2D( X, fig=fig, axs=axs,
        radius=R, color='cornflowerblue' ).draw()

    # Axis setup.
    plt.axis( Abound*np.array( [-1, 1, -1, 1] ) )
    plt.gca().set_aspect( 'equal', adjustable='box' )
    plt.xticks( [-i for i in range( -Abound,Abound+1 )] )
    plt.yticks( [-i for i in range( -Abound,Abound+1 )] )
    plt.show( block=0 )

    # Simulation block.
    T = 10;  Nt = round( T/dt ) + 1
    for i in range( Nt ):
        for x in X.T:
            D = anchorMeasure( x[:,None], aList=A )
            Dx = anchorMeasure( x[:,None], aList=Ax )
            Dy = anchorMeasure( x[:,None], aList=Ay )

    input( "Press ENTER to exit program... " )
