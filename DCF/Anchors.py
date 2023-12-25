# System imports.
import argparse
import sys
from os.path import expanduser
sys.path.insert(0, expanduser('~')+'/prog/kman')

import numpy as np
from KMAN.Regressors import *

def rotz(theta):
    R = np.array( [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)] ] )
    return R

def circ(R, tList):
    Nth = tList.shape[0]
    cList = np.empty( (2,Nth) )
    for i, t in enumerate( tList ):
        cList[:,i] = np.array( [R*np.cos( t ), R*np.sin( t )] ).reshape( 2, )
    return cList

def noise(eps=1e-3, shape=(2,1)):
    return 2*eps*np.random.rand( shape[0], shape[1] ) - eps

def noiseCirc(eps=1e-3, N=1):
    y = np.empty( (2,N) )
    for i in range( N ):
        t = 2*np.pi*np.random.rand( 1, )
        R = eps*np.random.rand()
        y[:,i] = circ( R, t )[:,0]
    return y

def distance(x, y):
    d = np.sqrt( (x - y).T@(x - y) )
    return d.reshape(1,1)

class Anchors:
    def __init__(self, Aset, eps=None):
        # Anchor set initializations and dimensions.
        self.n, self.p = Aset.shape
        self.Aset = Aset

        # Measurement noise parameter.
        self.eps = eps

    def linearDiff(self):
        # Return linear difference matrix.
        A = -2*np.vstack( ( [
            [ap - aq for aq in self.Aset.T]
                for ap in self.Aset.T ] ) )
        return A

    def squareDiff(self, m=1):
        # Return squared-difference matrix.
        b = np.vstack( [
            [ap@ap[:,None] - aq@aq[:,None] for aq in self.Aset.T]
                for ap in self.Aset.T ] )
        B = np.kron( b, np.ones( (1,m) ) )
        return B

    def distanceSet(self, X):
        # Return distances of each point in X from each anchor in Aset.
        Dset = np.hstack( [
            np.vstack( [distance( x[:,None], a[:,None] ) for a in self.Aset.T] )
                for x in X.T] )

        # Incorporate noise if eps is not None.
        if self.eps is not None:
            Dset = Dset + noise( eps=self.eps, shape=Dset.shape )
            Dset[Dset < 0] = 0  # Distance value cannot be negative.

        # Return distance set.
        return Dset

class DistanceCoupledFunctions( Anchors ):
    def __init__(self, Aset, Xeq=None, m=1, eps=None):
        # Initialize anchor set variable.
        Anchors.__init__( self, Aset, eps=eps )

        # Desired equilibrium positions.
        self.m = m
        self.Xeq = self.A[:,:m] if Xeq is None else Xeq

        # Calculate relevant DCF matrices.
        self.A = self.linearDiff()
        self.B = self.squareDiff( m=m )
        self.K = self.resolutionMatrix()

    def resolutionMatrix(self, A=None):
        # Set linear difference matrix.
        A = self.A if A is None else A
        # Solve regression (via SVD) for (A^T A)^-1.
        K = (Regressor( A.T@A, np.eye( self.n,self.n ) ).dmd()[0])@A.T
        return K

    def distanceStack(self, Dset):
        # Stack distance-coupled measurements.
        h = np.vstack( [[dp**2 - dq**2 for dq in Dset] for dp in Dset] )
        # Return stack.
        return h

    def position(self, Dset):
        # Form squared distance stack.
        H = self.distanceStack( Dset )
        # Return distance-coupled position.
        return self.K@(H - self.B)

    def control(self, C, Dset):
        # Get distance-coupled position.
        Y = self.position( Dset )
        # Return distance-coupled control.
        return C@(Y - self.Xeq)
