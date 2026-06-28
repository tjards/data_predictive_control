'''
Implementation of convex Model Predictive Control (MPC)

Stardard Form:

x(k+1) = Ax(k) + Bu(k)
'''

import numpy as np

class MPC():

    def __init__(self, A, B, Q, R, P, x0, u0, h, m, constraints):

        # store the attributes passed in 
        self.A = np.array(A, ndmin=2)           # state matrix  
        self.B = np.array(B, ndmin=2)           # input matrix
        self.Q = np.array(Q, ndmin=2)           # state cost matrix
        self.R = np.array(R, ndmin=2)           # input cost matrix
        self.P = np.array(P, ndmin=2)           # terminal cost matrix
        self.x0 = np.array(x0).reshape(-1, 1)   # initial state
        self.u0 = np.array(u0).reshape(-1, 1)   # initial input
        self.h = h                              # prediction horizon
        self.m = m                              # control horizon (inputs frozen after m steps: m <= h)
        self.nx = self.A.shape[0]               # state dimensions
        self.nu = self.B.shape[1]               # input dimensions
        self.constraints = constraints

