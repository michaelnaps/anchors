
from anchors import *


# True system values.
A = 2
dt = 0.01
Nx = 2
Nu = 2
# C = np.eye( Nx )
C = np.array( [
    [2, 0],
    [0, 1]
] )


# Anchor values.
Na = np.random.randint(2, 100)
# Na = 4
q = 2*A*np.random.rand( 2,1 ) - A

print( 'number of anchors: ', Na )
print( 'desired position: ', q.T )


# Anchor and reflection sets.
aList = np.random.rand( 2,Na )
# aList = np.array( [
#     [1, 1, -1, -1],
#     [1, -1, -1, 1]
# ] )
rxList = np.vstack( (-aList[0], aList[1]) )
ryList = np.vstack( (aList[0], -aList[1]) )


# Control matrices.
D = 1/4*np.diag( [ 1/np.sum( aList[0] ), 1/np.sum( aList[1] ) ] )
Q = np.sum( 4*q*aList, axis=1 )[:,None]


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


# Control methods of interest.
def control(x, q=[[0],[0]]):
    return C@(q - x)

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
    return C@D@(d + Q)


# Main execution block.
if __name__ == '__main__':
    x0 = np.random.rand( 2,1 )

    print( 'ideal control: ', control( x0, q=q ).T )
    print( 'anchor control:', anchorControl( x0 ).T )