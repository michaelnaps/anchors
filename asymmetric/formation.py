import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *


# Set hyper parameter(s).
Nr = 3                      # Number of anchor sets + reflection sets.
N = 1                       # Number of anchors.
M = 10                      # Number of vehicles.


# Exclusion elements in measurement function.
def exclude(i, j):
    sym = N*Nr - 1
    return i > sym


# Anchor set.
A = np.array( [[3],[4]] )
print( 'A:\n', A )

# Asymmetric vehicle set.
B = noise( eps=Abound, shape=(Nx,M-N*Nr) )

# Reflection sets.
Ax = Rx@A
Ay = Ry@A
Q = np.hstack( (A, Ax, Ay, B) )
print( 'Ax:\n', Ax )
print( 'Ay:\n', Ay )
print( 'Q:\n', Q )

# For error calculation.
Qerr = np.vstack( (Q, np.ones( (1,M) )) )

# Calculate anchor coefficient matrices.
S = signedCoefficientMatrix( N )
Z = anchorCoefficientMatrix( A, N, exclude=exclude )
print( 'S:\n', S )
print( 'Z:\n', Z )


# Main execution block.
if __name__ == '__main__':
    # Simulation time.
    T = 5;  Nt = round( T/dt ) + 1
    tList = np.array( [ [i*dt for i in range( Nt )] ] )

    # Initialize vehicle positions.
    delta = 2.50
    eps = 5.00
    X = Q + noiseCirc( eps=delta, N=M )

    # Initial error calculation.
    regr = Regressor( Qerr, X )
    T, _ = regr.dmd();  e0 = np.vstack( (0, regr.err) )

    # Used for plotting without sim.
    xList = np.empty( (M,Nt,Nx) )
    eList = np.empty( (2,Nt) )
    xList[:,0,:] = X.T
    eList[:,0] = e0[:,0]

    # Initialize plot with vehicles, anchors and markers.
    fig, axs, swrm, anchors, error = initAnchorEnvironment(
        X, Q, A, e0, Nt=Nt, Np=2, R1=0.40, R2=delta)

    # Environment block.
    print( 'Xi: %0.3f\n' % regr.err, X )
    input( "Press ENTER to begin simulation..." )
    for i in range( Nt ):
        # Take measurements.
        H = anchorMeasure( X, X, eps=eps, exclude=exclude )**2

        # Calculate control and add disturbance.
        U = C@(Q - (Z*S)@H[:N*Nr])
        if i > 200 and i < 300:
            W = 10.0
            P = 0
            U[:,:P] = U[:,:P] - W*np.ones( (Nx,P) )

        # Apply dynamics.
        X = model( X, U )

        # Calculate tranformation error.
        regr = Regressor( Qerr, X )
        T, _ = regr.dmd()

        # Save values.
        eList[:,i] = np.array( [dt*i, regr.err] )
        xList[:,i,:] = X.T

        # Update simulation.
        if sim and i % n == 0:
            swrm.update( X )
            error.update( eList[:,i,None] )
            # axs[1].set_title( 'time: %s' % i )
            plt.pause( pausesim )
    print( 'Xf:\n', X )

    # Calculate transformation matrix by DMD.
    Qerr = np.vstack( (Q, np.ones( (1,M) )) )
    regr = Regressor( Qerr, X )
    T, _ = regr.dmd()
    print( 'T:\n', T )

    # Plot transformed grid for reference.
    if not sim:
        swrm.update( X )
        for vhc in xList:
            axs[0].plot( vhc.T[0], vhc.T[1], color='cornflowerblue' )
        axs[1].plot( eList[0], eList[1], color='cornflowerblue' )
    anchors.update( T@Qerr )
    xaxis = T@np.array( [[-Abound, Abound],[0, 0],[1, 1]] )
    yaxis = T@np.array( [[0, 0],[-Abound, Abound],[1, 1]] )
    axs[0].plot( xaxis[0], xaxis[1], color='grey', linestyle='--' )
    axs[0].plot( yaxis[0], yaxis[1], color='grey', linestyle='--' )
    plt.pause( pausesim )

    # Calculate error after transformation.
    print( '\nError: ', np.linalg.norm( X - T@Qerr ) )
    input( "Press ENTER to exit program..." )
