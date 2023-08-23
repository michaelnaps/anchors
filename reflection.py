
from anchors import *

# Hyper parameter(s)
A = 2
dt = 0.01
Nx = 2
Nu = 2
Na = 4
q = 2*A*np.random.rand( 2,1 ) - A


# Anchor and reflection sets.
aList = 2*A*np.random.rand( 2,Na ) - A


# Control matrices.
D = 1/2*np.diag( [
    1/np.sum( aList[0] ), 1/np.sum( aList[1] )
] )
Q = D@np.sum(
    2*q*aList - aList**2, axis=1
)[:,None]


# Reflection set.
def getReflectionSet( axis=0 ):
    rxList = np.zeros( (2,Na) )
    ryList = np.zeros( (2,Na) )
    rxList[1,:] = aList[1]
    ryList[0,:] = aList[0]
    return rxList, ryList


# Anchor measurement functions.
def anchorMeasure(x):
    d = np.empty( (1,Na) )
    for i, a in enumerate( aList.T ):
        d[:,i] = (x - a[:,None]).T@(x - a[:,None])
    return np.sqrt( d )

def reflectionMeasure(x, rList):
    dr = np.empty( (1,Na) )
    for i, r in enumerate( rList.T ):
        dr[:,i] = (x - r[:,None]).T@(x - r[:,None])
    return np.sqrt( dr )


# Anchor-based control policy.
def anchorControl(x):

    # Combine anchor sets.
    dList = anchorMeasure( x )
    rList = reflectionMeasure( x )

    # Calculate measurement state.
    d = np.array( [
        np.sum( dList**2 - rList[0]**2, axis=1 ),
        np.sum( dList**2 - rList[1]**2, axis=1 )
    ] )

    # Return control.
    return D@d + Q


# Main execution block.
if __name__ == '__main__':
    x0 = np.random.rand( 2,1 )

    print( 'ideal control: ', control( x0, q=q ).T )
    print( 'anchor control:', anchorControl( x0 ).T )