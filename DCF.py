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

    def anchorDifferenceMatrices(self, m=1):
        # Difference coefficient matrix.
        A = -2*np.vstack( ( [
            [ap - aq for aq in self.Aset.T]
                for ap in self.Aset.T ] ) )

        # Squared-difference matrix.
        b = np.vstack( [
            [ap@ap[:,None] - aq@aq[:,None] for aq in self.Aset.T]
                for ap in self.Aset.T ] )
        B = np.kron( b, np.ones( (1,m) ) )

        # Return matrices.
        return A, B

class DistanceCoupledFunctions( Anchors ):
    def __init__(self, Aset, m=1):
        # Initialize anchor set variable.
        self.anchors = Anchors( self, Aset )

        # Calculate relevant DCF matrices.
        self.A, self.B = self.anchors.anchorDifferenceMatrices()
        self.C, self.K = self.distanceCoupledMatrices()

    def distanceCoupledMatrices(self):
        Z, _ = Regressor( self.A.T@self.A, np.eye( Nx,Nx ) ).dmd()
        K = Z@self.A.T
        return K

    # def position(self, Dset=None):
    #     if Dset is None:
    #         # Calculate distances internally.
