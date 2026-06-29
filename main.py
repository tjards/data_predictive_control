
# standard inputs
import numpy as np
import json

# custom imports
import mpc 


# Load parameters from config
with open('config.json') as f:
    cfg = json.load(f)

# load parameters from config
A   = np.array(cfg['A'])
B   = np.array(cfg['B'])
Q   = np.diag(cfg['Q_diag'])
R   = np.diag(cfg['R_diag'])
P   = np.diag(cfg['P_diag'])
x0  = np.array(cfg['x0'])
u0  = np.array(cfg['u0'])
h   = cfg['h']
m   = cfg['m']
constraints = cfg['constraints']

# initialize the MPC controller
mpc = mpc.MPC(A, B, Q, R, P, x0, u0, h, m, constraints)


