'''
Implementation of convex Model Predictive Control (MPC)

Standard Form:

x(k+1) = Ax(k) + Bu(k)

Constraints:

    x_min <= x(k) <= x_max
    u_min <= u(k) <= u_max

    or 

    Mx <= bx
    Mu <= bu

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
        self._define_constraints()
        self._build_augmented_matrices()

    def _define_constraints(self):

        if self.constraints['type'] == 'box':
            self.x_min = np.array(self.constraints['x_min']).reshape(-1, 1)
            self.x_max = np.array(self.constraints['x_max']).reshape(-1, 1)
            self.u_min = np.array(self.constraints['u_min']).reshape(-1, 1)
            self.u_max = np.array(self.constraints['u_max']).reshape(-1, 1)
        elif self.constraints['type'] == 'lmi':
            self.Mx = np.array(self.constraints['Mx'], ndmin=2)
            self.bx = np.array(self.constraints['bx'], ndmin=2)
            self.Mu = np.array(self.constraints['Mu'], ndmin=2)
            self.bu = np.array(self.constraints['bu'], ndmin=2)
        else:
            raise ValueError("Invalid constraint type: Must be 'box' or 'lmi'")
        
    def _build_augmented_matrices(self):

        # initialize the augmented matrices over prediction horizon
        self.A_aug = np.zeros((self.nx * self.h, self.nx))
        self.B_aug = np.zeros((self.nx * self.h, self.nu * self.h))

        # build the augmented matrices
        for i in range(self.h):
            self.A_aug[i*self.nx:(i+1)*self.nx, :] = np.linalg.matrix_power(self.A, i+1)
            for j in range(i+1):
                self.B_aug[i*self.nx:(i+1)*self.nx, j*self.nu:(j+1)*self.nu] = np.linalg.matrix_power(self.A, i-j) @ self.B


    




