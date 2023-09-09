import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *


# Set hyper parameter(s).
Nr = 3                      # Number of anchor sets + reflection sets.
N = 3                       # Number of anchors.
# N = np.random.randint(1,10)
M = Nr*N                    # Number of vehicles.


# Exclusion elements in measurement function.
def exclude(i, j):
    return False # (j - i) % N == 0


# Anchor set.
A = np.array( [[2,5,8],[3,6,9]] )
# A = Abound*np.random.rand( 2,N )
# A = np.array( [
#     [i for i in range( int( Abound + 1 ) ) if i%2 != 0],
#     [i for i in range( int( Abound + 1 ) ) if i%2 != 0]] )
# A = noise( eps=Abound, shape=(2,N) )
print( 'A:\n', A )


# Reflection sets.
Ax = Rx@A
Ay = Ry@A
Q = np.hstack( (A, Ax, Ay) )
print( 'Ax:\n', Ax )
print( 'Ay:\n', Ay )
print( 'Q:\n', Q )

# For error calculation.
Qerr = np.vstack( (Q, np.ones( (1,M) )) )


# Signed coefficient matrix.
S = np.array( [
    np.hstack( ([1 for i in range( N )], [0 for i in range( N )], [-1 for i in range( N )]) ),
    np.hstack( ([1 for i in range( N )], [-1 for i in range( N )], [0 for i in range( N )]) ) ] )
print( 'S:\n', S )


# Anchor-position coefficients.
z = -1/4*np.hstack( [
    [ 1/np.sum( np.hstack( [A[:,j,None]
        for j in range( N ) if not exclude(i,j)] ), axis=1 )
            for i in range( N ) ] ] ).T
print( 'z:\n', z )
Z = np.hstack( (z, z, z) )
print( 'Z:\n', Z )


# Main execution block.
if __name__ == '__main__':
    # Simulation time.
    T = 5;  Nt = round( T/dt ) + 1
    tList = np.array( [ [i*dt for i in range( Nt )] ] )

    # Initialize vehicle positions.
    delta = 2.50
    eps = 0.0
    X = Q + noiseCirc( eps=delta, N=M )

    # Initial error calculation.
    regr = Regressor( Qerr, X )
    T, _ = regr.dmd();  e0 = np.vstack( (0, regr.err) )

    # Used for plotting without sim.
    xList = np.empty( (M,Nt,Nx) )
    eList = np.empty( (2,Nt) )
    xList[:,0,:] = X.T
    eList[:,0] = e0[:,0]

    # Swarm variables.
    Np = 2
    R = 0.40
    fig, axs = plt.subplots(1,Np)
    swrm = Swarm2D( X, fig=fig, axs=axs[0], zorder=100,
        radius=-R, color='cornflowerblue', draw_tail=sim
        ).draw()
    anchors = Swarm2D( Q, fig=fig, axs=axs[0], zorder=50,
        radius=R, draw_tail=False, color='indianred'
        ).setLineStyle( None, body=True ).draw()
    disturb = Swarm2D( Q, fig=fig, axs=axs[0], zorder=10,
        radius=delta, draw_tail=False, color='none'
        ).setLineStyle( ':', body=True
        ).setLineWidth( 1.0, body=True ).draw()
    axs[0].plot( Q[0], Q[1], zorder=50, color='indianred',
        linestyle='none', marker='x' )
    axs[0].plot( X[0], X[1], zorder=50, color='cornflowerblue',
        linestyle='none', marker='x' )

    # For plotting error.
    error = Vehicle2D( e0, fig=fig, axs=axs[1],
        radius=0.0, color='cornflowerblue', tail_length=Nt ).draw()
    axs[1].plot( [0, dt*Nt], [0, 0], color='indianred', linestyle='--' )

    # Axis setup.
    bounds = np.vstack( [
        1.5*Abound*np.array( [-1, 1, -1, 1] ),
        np.hstack( [e0[0], dt*Nt, -0.5, e0[1]] ) ] )
    for a, bnd in zip( axs, bounds ):
        a.axis( bnd )
        a.grid( 1 )
    axs[0].axis( 'equal' )
    plt.show( block=0 )

    # Environment block.
    print( 'Xi: %0.3f\n' % regr.err, X )
    input( "Press ENTER to begin simulation..." )
    for i in range( Nt ):
        # Take measurements.
        H = anchorMeasure( X, X, eps=eps, exclude=exclude )**2

        # Calculate control and add disturbance.
        U = C@(Q - (Z*S)@H)
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
            plt.pause( sim_pause )
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
    plt.pause( sim_pause )

    # Calculate error after transformation.
    print( '\nError: ', np.linalg.norm( X - T@Qerr ) )
    input( "Press ENTER to exit program..." )
