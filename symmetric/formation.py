import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/anchors')

from root import *


# Set hyper parameter(s).
Nr = 3                      # Number of anchor sets + reflection sets.
# N = np.random.randint(1,8)  # Number of anchors.
N = 4
M = Nr*N                    # Number of vehicles.


# Exclusion elements in measurement function.
def exclude(i, j):
    return (j - i) % N == 0


# Anchor set.
A = np.array( [[0,2,5,8],[0,3,6,9]] )
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
B = np.array( [
    np.hstack( ([1 for i in range( N )], [0 for i in range( N )], [-1 for i in range( N )]) ),
    np.hstack( ([1 for i in range( N )], [-1 for i in range( N )], [0 for i in range( N )]) ) ] )
print( 'B:\n', B )


# Anchor-position coefficients.
z = -1/4*np.hstack( [
    [ 1/np.sum( np.hstack( [A[:,j,None]
        for j in range( N ) if i != j] ), axis=1 )
            for i in range( N ) ] ] ).T
print( 'z:\n', z )
Z = np.hstack( (z, z, z) )
print( 'Z:\n', Z )


# Main execution block.
if __name__ == '__main__':
    # Simulation time.
    T = 10;  Nt = round( T/dt ) + 1
    tList = np.array( [ [i*dt for i in range( Nt )] ] )

    # Initialize vehicle positions.
    delta = 5.0
    eps = 0.0
    X = np.hstack( (
        A + noiseCirc( eps=delta, N=N ),
        Ax + noiseCirc( eps=delta, N=N ),
        Ay + noiseCirc( eps=delta, N=N ) ) )
    print( 'Xi:\n', X )

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

    # For plotting error.
    error = Vehicle2D( np.array( [[0],[0]] ), fig=fig, axs=axs[1],
        radius=0.01, color='cornflowerblue', tail_length=Nt ).draw()

    # Axis setup.
    for a in axs:
        a.axis( 'equal' )
        a.axis( 1.5*Abound*np.array( [-1, 1, -1, 1] ) )
        a.grid( 1 )
    plt.show( block=0 )

    # Environment block.
    input( "Press ENTER to begin simulation..." )
    for t in tList[0]:
        # Take measurements.
        H = anchorMeasure( X, X, eps=eps, exclude=exclude )**2

        # Calculate control and add disturbance.
        U = C@(Q - Z*(B@H))
        if t > 250 and t < 500:
            W = 2.5
            P = 1
            U[:,:P] = U[:,:P] + W*np.ones( (Nx,P) )

        # Apply dynamics.
        X = model( X, U )

        # Calculate tranformation error.
        regr = Regressor( Qerr, X )
        T, _ = regr.dmd()

        # Update simulation.
        if sim and round( t/dt ) % n == 0:
            swrm.update( X )
            error.update( np.vstack( (t, regr.err) ) )
            # axs[1].set_title( 'time: %s' % t )
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
    anchors.update( T@Qerr )
    xaxis = T@np.array( [[-Abound, Abound],[0, 0],[1, 1]] )
    yaxis = T@np.array( [[0, 0],[-Abound, Abound],[1, 1]] )
    axs[0].plot( xaxis[0], xaxis[1], color='grey', linestyle='--' )
    axs[0].plot( yaxis[0], yaxis[1], color='grey', linestyle='--' )
    plt.pause( sim_pause )

    # Calculate error after transformation.
    print( '\nError: ', np.linalg.norm( X - T@Qerr ) )
    # ans = input( "Press ENTER to see error plot..." )
    # if ans == 'n':
    #     exit()

    # # Plot error vs. iterations.
    # figErr, axsErr = plt.subplots()
    # axsErr.plot( tList[0], eList[0] )
    # plt.show( block=0 )

    input( "Press ENTER to exit program..." )
