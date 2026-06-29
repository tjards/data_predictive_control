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
import cvxpy as cp

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
        self.m = m                              # FOR LATER: control horizon (inputs frozen after m steps: m <= h)
        self.nx = self.A.shape[0]               # state dimensions
        self.nu = self.B.shape[1]               # input dimensions
        self.constraints = constraints
        
        self.update_internal_parameters()  

    # allows for updated parameters
    def update_internal_parameters(self):

        self._define_constraints()
        self._augment_matrices()
        self._construct_optimization_problem()

    # constraints can be defined in different ways 
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

    # augment all the things 
    def _augment_matrices(self):
        self._build_augmented_system_matrices()
        self._build_augmented_cost_matrices()
        self.build_augmented_constraints()

    def _build_augmented_system_matrices(self):

        # initialize the augmented matrices over prediction horizon
        self.A_aug = np.zeros((self.nx * self.h, self.nx))
        self.B_aug = np.zeros((self.nx * self.h, self.nu * self.h))

        # build the augmented matrices
        for i in range(self.h):
            self.A_aug[i*self.nx:(i+1)*self.nx, :] = np.linalg.matrix_power(self.A, i+1)
            for j in range(i+1):
                self.B_aug[i*self.nx:(i+1)*self.nx, j*self.nu:(j+1)*self.nu] = np.linalg.matrix_power(self.A, i-j) @ self.B

    def _build_augmented_cost_matrices(self):

        # initialize the augmented cost matrices over prediction horizon
        self.Q_aug = np.zeros((self.nx * self.h, self.nx * self.h))
        self.R_aug = np.zeros((self.nu * self.h, self.nu * self.h))

        # build the augmented cost matrices
        for i in range(self.h):
            # up to but not including the terminal step, use Q 
            if i < self.h - 1:
                self.Q_aug[i*self.nx:(i+1)*self.nx, i*self.nx:(i+1)*self.nx] = self.Q
            # at terminal step (i.e., end of prediction horizon), use P
            else:
                self.Q_aug[i*self.nx:(i+1)*self.nx, i*self.nx:(i+1)*self.nx] = self.P

            # R is used for all steps 
            self.R_aug[i*self.nu:(i+1)*self.nu, i*self.nu:(i+1)*self.nu] = self.R
    
    def build_augmented_constraints(self):
       
        if self.constraints["type"] == "box":
            self.x_min_aug = np.tile(np.asarray(self.x_min).reshape(-1), self.h)
            self.x_max_aug = np.tile(np.asarray(self.x_max).reshape(-1), self.h)
            self.u_min_aug = np.tile(np.asarray(self.u_min).reshape(-1), self.h)
            self.u_max_aug = np.tile(np.asarray(self.u_max).reshape(-1), self.h)

        elif self.constraints["type"] == "lmi":
            self.Mx_aug = np.kron(np.eye(self.h), self.Mx)
            self.bx_aug = np.tile(np.asarray(self.bx).reshape(-1), self.h)
            self.Mu_aug = np.kron(np.eye(self.h), self.Mu)
            self.bu_aug = np.tile(np.asarray(self.bu).reshape(-1), self.h)

        else:
            raise ValueError(f"Unknown constraint type: {self.constraints['type']}")
        
    def _construct_optimization_problem(self):

        # define the optimization variables
        self.opt_u      = cp.Variable((self.nu * self.h, 1))
        self.opt_cost   = cp.quad_form(self.A_aug @ self.x0 + self.B_aug @ self.opt_u, self.Q_aug) + cp.quad_form(self.opt_u, self.R_aug)
        self.opt_constraints = []

        if self.constraints["type"] == "box":
            self.opt_constraints += [self.x_min_aug <= self.A_aug @ self.x0 + self.B_aug @ self.opt_u]
            self.opt_constraints += [self.A_aug @ self.x0 + self.B_aug @ self.opt_u <= self.x_max_aug]
            self.opt_constraints += [self.u_min_aug <= self.opt_u]
            self.opt_constraints += [self.opt_u <= self.u_max_aug]

        elif self.constraints["type"] == "lmi":
            self.opt_constraints += [self.Mx_aug @ (self.A_aug @ self.x0 + self.B_aug @ self.opt_u) <= self.bx_aug]
            self.opt_constraints += [self.Mu_aug @ self.opt_u <= self.bu_aug]

        else:
            raise ValueError(f"Unknown constraint type: {self.constraints['type']}")

        # define the optimization problem
        self.prob = cp.Problem(cp.Minimize(self.opt_cost), self.opt_constraints)

    def solve(self, x0, u0):

        self.x0 = np.array(x0).reshape(-1, 1)
        self.u0 = np.array(u0).reshape(-1, 1)

        # (re)build the problem with current x0,  u0
        self._construct_optimization_problem()

        # solve the optimization problem
        self.prob.solve()

        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MPC solver failed: {self.prob.status}")

        # results 
        self.result_control_next = self.opt_u.value[:self.nu]
        self.result_control_sequence = self.opt_u.value
        self.result_state_sequence = self.A_aug @ self.x0 + self.B_aug @ self.opt_u.value







