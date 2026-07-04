'''
Implementation of convex Model Predictive Control (MPC) with disturbance rejection

Standard Form:

x(k+1) = Ax(k) + B (u(k) + d) 
J = sum_{i=0}^{h-1} x(k+i)'Qx(k+i) + u(k+i)'Ru(k+i) + x(k+h)'Px(k+h)

Constraints:

    x_min <= x(k) <= x_max
    u_min <= u(k) <= u_max

    or 

    Mx <= bx
    Mu <= bu

'''

import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_are
#from dataclasses import dataclass
import json


# load parameters from config
with open('configs/config_mpc.json') as f:
    cfg = json.load(f)

Ts = cfg['Ts']  
Tf = cfg['Tf']
A   = 0*np.array(cfg['A'])
B   = 0*np.array(cfg['B'])
Q   = np.diag(cfg['Q_diag'])
R   = np.diag(cfg['R_diag'])
P   = cfg['P_diag']  # may be 'dare' string or a list of diagonal values
#x0  = np.array(cfg_mpc['x0'], dtype=float)
u0  = np.array(cfg['u0'], dtype=float)
h   = cfg['h']
m   = cfg['m']
constraints = cfg['constraints']
disturbance = cfg['disturbance']
learning_rate = cfg['learning_rate']
window_size = cfg['window_size']
optimizer = cfg['optimizer']
update_parameters_rate = cfg['update_parameters_rate']

# online modeller
class Modeller():
    
    def __init__(self):

        self.A_hat = np.array(A, ndmin=2)
        self.B_hat = np.array(B, ndmin=2)
        self.nx = int(A.shape[0])
        self.nu = int(B.shape[1])
        self.window_size = int(window_size)         # window to collect data                # refit every this many steps
        #self.A_hat = np.zeros((self.nx, self.nx))
        #self.B_hat = np.zeros((self.nx, self.nu))
        self.Phi = np.zeros((self.nx, self.nx + self.nu))  
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.update_parameters_rate = int(update_parameters_rate) 
        self.update_parameters_count = 0

        self.X = np.zeros((self.nx, self.window_size))
        self.U = np.zeros((self.nu, self.window_size))

        self.scale_x  = np.ones((self.nx, 1))
        self.scale_u  = np.ones((self.nu, 1))
        self.scale_dx = np.ones((self.nx, 1))

        self.viable = False
        self.shared = False

        self.Ts = Ts
        self.constraints = constraints
        self.u0 = u0

    def excite(self, plant, x, state_history, input_history):

        model_log = open('model_log.txt', 'w')
        model_log.write('step,A_hat,B_hat\n') # headers

        u_max          = np.array(self.constraints['u_max'])
        u_min          = np.array(self.constraints['u_min'])
        excite_hold    = 0.2                                   # seconds to hold each random input
        excite_hold_steps = int(excite_hold / self.Ts)               # steps per hold
        excite_max_steps  = int(15 / self.Ts)                       # max excitation duration (seconds)
        rng = np.random.default_rng(seed=42)
        found_stable_model = False
        k = 0
        u_exc = np.zeros_like(u_min)    # will be set on first transition
        excite_count = excite_hold_steps # trigger immediate draw on first step
        rockback = False

        while k < excite_max_steps:

            if excite_count >= excite_hold_steps and not rockback:
                excite_count = 0
                u_exc = rng.uniform(u_min, u_max)
                rockback = True
            elif excite_count >= excite_hold_steps and rockback:
                excite_count = 0
                u_exc = -u_exc
                rockback = False
            excite_count += 1

            self.update(x, u_exc)
            x = plant.evolve(x, u_exc, disturb = False)
                    
            state_history.append(x.copy())
            input_history.append(u_exc.copy())
            model_log.write(f'exc_{k + 1},{self.A_hat.tolist()},{self.B_hat.tolist()}\n')
            
            k += 1

        model_log.close()
        
        return x, state_history, input_history




    def update(self, x, u):

        self.accumulate_data(x, u)
        self.update_parameters_count += 1
        if self.update_parameters_count >= self.update_parameters_rate:
            self.fit()
            self.update_parameters_count = 0
            self.do_fit = True
        else:
            self.do_fit = False
    # accumulate data over window 
    def accumulate_data(self, x, u):

        # shift data to the left and add new data to the right
        self.X[:, :-1] = self.X[:, 1:]
        self.U[:, :-1] = self.U[:, 1:]
        self.X[:, -1] = x.flatten()
        self.U[:, -1] = u.flatten()

    # fit x(k+1) = Ix(k) + A_tilde*x(k) + B*u(k) + Bd
    # equivalently: dx = x(k+1) - x(k) = A_tilde*x(k) + B*u(k) + bias
    def fit(self):

        if self.optimizer == 'least_squares':

            X_curr = self.X[:, :-1]
            U_curr = self.U[:, :-1]
            X_next = self.X[:, 1:]
            N = X_curr.shape[1]

            # factor out identity: regress on the residual
            dX = X_next - X_curr                                                # (nx, N)

            # per-feature scale factors (std over window, floored to avoid divide-by-zero)
            self.scale_x  = np.maximum(np.std(X_curr, axis=1, keepdims=True), 1e-8)  # (nx, 1)
            self.scale_u  = np.maximum(np.std(U_curr, axis=1, keepdims=True), 1e-8)  # (nu, 1)
            self.scale_dx = np.maximum(np.std(dX,     axis=1, keepdims=True), 1e-8)  # (nx, 1)

            # normalize
            X_n  = X_curr / self.scale_x
            U_n  = U_curr / self.scale_u
            dX_n = dX     / self.scale_dx

            # stack normalized regressor with bias column (absorbs Bd in normalized space)
            Z_n = np.vstack([X_n, U_n, np.ones((1, N))])

            # least squares in normalized space: dX_n ≈ Phi_n * Z_n
            Phi_n, _, _, _ = np.linalg.lstsq(Z_n.T, dX_n.T, rcond=None)
            Phi_n = Phi_n.T                                                     # (nx, nx+nu+1)

            # un-normalize: A_tilde = diag(scale_dx) @ Phi_n[:, :nx] @ diag(1/scale_x)
            S_dx = np.diag(self.scale_dx.flatten())
            A_tilde = S_dx @ Phi_n[:, :self.nx]          @ np.diag(1.0 / self.scale_x.flatten())
            B_new   = S_dx @ Phi_n[:, self.nx:self.nx+self.nu] @ np.diag(1.0 / self.scale_u.flatten())

            # recover A_hat by adding back the identity
            A_new = np.eye(self.nx) + A_tilde

            self.A_hat = (1-self.learning_rate) * self.A_hat + self.learning_rate * A_new
            self.B_hat = (1-self.learning_rate) * self.B_hat + self.learning_rate * B_new

            # mark as stable and controllable when both conditions hold
            is_stable = np.all(np.abs(np.linalg.eigvals(self.A_hat)) < 1.1)

            # controllability matrix [B | AB | A^2 B | ... | A^(nx-1) B]
            C = np.hstack([np.linalg.matrix_power(self.A_hat, i) @ self.B_hat
                           for i in range(self.nx)])
            is_controllable = np.linalg.matrix_rank(C) == self.nx

            if is_stable and is_controllable:
                self.viable = True
                print('stable and controllable model found')
            else:
                self.viable = False

            self.shared = False

        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

# controller
class MPC():

    def __init__(self, x0):

        # store the attributes passed in 
        self.A = np.array(A, ndmin=2)           # state matrix  
        self.B = np.array(B, ndmin=2)           # input matrix
        self.Q = np.array(Q, ndmin=2)           # state cost matrix
        self.R = np.array(R, ndmin=2)           # input cost matrix
        
        # we can compute terminal cost based on solution to Discrete Algebraic Riccati Equation
        if isinstance(P, str) and P.lower() == 'dare':
            self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)  
        # or hardcoded
        else:
            self.P = np.array(P, ndmin=2)       # terminal
        self.x0 = np.array(x0).reshape(-1, 1)   # initial state
        self.u0 = np.array(u0).reshape(-1, 1)   # initial input
        self.h = h                              # prediction horizon
        self.m = m                              # control horizon (inputs freeze after m <= h)
        self.nx = self.A.shape[0]               # state dimensions
        self.nu = self.B.shape[1]               # input dimensions
        self.constraints = constraints
        self.disturbance = disturbance
        self.Ts = Ts
        self.Tf = Tf

        # we can do disturbance rejection
        if self.disturbance:
            self.d_hat = np.zeros((self.nu, 1))  # estimated input disturbance
            self.x_prev = None                   # previous measured state
            self.u_prev = None                   # previous commanded input
        
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
        self._build_augmented_constraints()

    def _build_augmented_system_matrices(self):

        # initialize the augmented matrices over prediction horizon
        self.A_aug = np.zeros((self.nx * self.h, self.nx))
        self.B_aug = np.zeros((self.nx * self.h, self.nu * self.h))
        if self.disturbance:
            self.D_aug = np.zeros((self.nx * self.h, self.nu * self.h))
        else:
            self.D_aug = None

        # build the augmented matrices
        for i in range(self.h):
            self.A_aug[i*self.nx:(i+1)*self.nx, :] = np.linalg.matrix_power(self.A, i+1)
            for j in range(i+1):
                self.B_aug[i*self.nx:(i+1)*self.nx, j*self.nu:(j+1)*self.nu] = np.linalg.matrix_power(self.A, i-j) @ self.B
        if self.disturbance:
            self.D_aug = self.B_aug @ np.tile(np.eye(self.nu), (self.h, 1))

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
    
    def _build_augmented_constraints(self):
       
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

        # define the optimization variables (optimized up to m, then will be frozen up to h)
        self.opt_u = cp.Variable((self.nu * self.m, 1))

        # define full input sequence: opt_u up to m + frozen inputs 
        if self.m < self.h:
            u_last = self.opt_u[(self.m - 1) * self.nu : self.m * self.nu]
            self.u_full = cp.vstack([self.opt_u] + [u_last] * (self.h - self.m))
        else:
            self.u_full = self.opt_u

        # disturbance feedforward offset
        if self.disturbance:
            d_offset = self.D_aug @ self.d_hat
        else:
            d_offset = np.zeros((self.nx * self.h, 1))

        # define the predicted state sequence     
        x_pred = self.A_aug @ self.x0 + self.B_aug @ self.u_full + d_offset

        # define the cost function 
        self.opt_cost = cp.quad_form(x_pred, self.Q_aug) + cp.quad_form(self.u_full, self.R_aug)
        
        #define the constraints
        self.opt_constraints = []
        if self.constraints["type"] == "box":
            self.opt_constraints += [self.x_min_aug <= x_pred]
            self.opt_constraints += [x_pred <= self.x_max_aug]
            self.opt_constraints += [self.u_min_aug <= self.u_full]
            self.opt_constraints += [self.u_full <= self.u_max_aug]
        elif self.constraints["type"] == "lmi":
            self.opt_constraints += [self.Mx_aug @ x_pred <= self.bx_aug]
            self.opt_constraints += [self.Mu_aug @ self.u_full <= self.bu_aug]
        else:
            raise ValueError(f"Unknown constraint type: {self.constraints['type']}")

        # define the optimization problem
        self.prob = cp.Problem(cp.Minimize(self.opt_cost), self.opt_constraints)

    def solve(self, x0, u0):

        self.x0 = np.array(x0).reshape(-1, 1)
        self.u0 = np.array(u0).reshape(-1, 1)

        # update disturbance estimate from one-step prediction error
        if self.disturbance and self.x_prev is not None:
            prediction_error = self.x0 - self.A @ self.x_prev - self.B @ self.u_prev
            self.d_hat, _, _, _ = np.linalg.lstsq(self.B, prediction_error, rcond=None)

        # (re)build the problem with current x0, u0, d_hat
        self._construct_optimization_problem()

        # solve the optimization problem
        self.prob.solve()

        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"MPC solver failed: {self.prob.status}")

        # results
        self.result_control_next        = self.u_full.value[:self.nu]
        self.result_control_sequence    = self.u_full.value
        if self.disturbance:
            d_offset                    = self.D_aug @ self.d_hat
        else:
            d_offset                    = np.zeros((self.nx * self.h, 1))
        self.result_state_sequence      = self.A_aug @ self.x0 + self.B_aug @ self.u_full.value + d_offset

        # store state and commanded input for next disturbance update
        if self.disturbance:
            self.x_prev = self.x0.copy()
            self.u_prev = self.result_control_next.copy()


