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

class Anchors:
    def __init__(self, Aset):
        self.n, self.p = Aset.shape
        self.Aset = Aset

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


class DistanceCoupledFunctions( Anchors ):
    def __init__(self, Aset, Xeq=None, m=1):
        # Initialize anchor set variable.
        self.anchors = Anchors( self, Aset )

        # Desired equilibrium positions.
        self.Xeq = self.A[:,:m] if Xeq is None else Xeq

        # Calculate relevant DCF matrices.
        self.A = self.anchors.linearDiff()
        self.B = self.anchors.squareDiff( m=m )
        self.K = self.resolutionMatrix()

    def resolutionMatrix(self, A=None):
        # Set linear difference matrix.
        A = self.A if A is None else A
        # Solve regression (via SVD) for (A^T A)^-1.
        K = (Regressor( A.T@A, np.eye( Nx,Nx ) ).dmd()[0])@A
        return K

    def position(self, Dset):
        # Form squared distance stack.
        H = self.distanceStack( Dset )
        # Return distance-coupled position.
        return self.K@(H - self.B)

    def control(self, Dset, C):
        # Get distance-coupled position.
        Y = self.position( Dset )
        # Return distance-coupled control.
        return C@(Y - self.Xeq)
