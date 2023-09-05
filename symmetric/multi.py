import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *


# Anchor values.
N = 2    # Number of anchors.
M = 3*N  # Number of vehicles.

# Anchor and corresponding reflection sets.
# A = Abound*np.random.rand( 2,N )
A = np.array( [[3, 5],[3, 5]] )
Ax = Rx@A
Ay = Ry@A
Amega = np.hstack( (A, Ax, Ay) )
print( 'A:\n', A )
print( 'Ax:\n', Ax )
print( 'Ay:\n', Ay )

# # Anchor-distance control matrices.
# D = -1/4*np.diag( [ 1/np.sum( A[0] ), 1/np.sum( A[1] ) ] )
# print( 'Anchor coefficient matrices:\n', D )


# Anchor measurement functions.
def anchorMeasure(x, aList=None, exclude=[-1]):
    if aList is None:
        aList = A
    Na = aList.shape[1]
    d = np.zeros( (1,Na) )
    for i, a in enumerate( aList.T ):
        if i not in exclude:
            d[:,i] = (x - a[:,None]).T@(x - a[:,None])
    return np.sqrt( d )


# Main execution block.
if __name__ == '__main__':
    # Initialize vehicle positions.
    offset = 10
    X = np.hstack( (
        A + noise( eps=offset, shape=(2,N) ),
        Ax + noise( eps=offset, shape=(2,N) ),
        Ay + noise( eps=offset, shape=(2,N) ),
    ) )
    print( 'X:\n', X.T )

    # Initialize control sets.
    k = 0
    zSet = np.empty( (M,2,2) )
    qSet = np.empty( (M,2,1) )
    for i, x in enumerate( X.T ):
        if k == N:  k = 0
        Atemp = np.array( [A[:,j] for j in range( N ) if k != j] ).T
        zSet[i] = -1/4*np.diag( [ 1/np.sum( Atemp[0] ), 1/np.sum( Atemp[1] ) ] )
        qSet[i] = Amega[:,i,None]
        k += 1
    print( 'Z:\n', zSet )
    print( 'Q:\n', qSet )

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
    for t in range( Nt ):
        j = 0
        for i, (x, q, Z) in enumerate( zip( X.T, qSet, zSet ) ):
            if j == N:  j = 0
            D = anchorMeasure( x[:,None], aList=A, exclude=[j] )
            Dx = anchorMeasure( x[:,None], aList=Ax, exclude=[j] )
            Dy = anchorMeasure( x[:,None], aList=Ay, exclude=[j] )
            z = np.vstack( (
                np.sum( D**2 - Dy**2, axis=1 ),
                np.sum( D**2 - Dx**2, axis=1 )
            ) )
            X[:,i] = model( x[:,None], C@(q - Z@z) )[:,0]
            j += 1
        swrm.update( X )
        plt.pause( 1e-3 )


    input( "Press ENTER to exit program... " )
