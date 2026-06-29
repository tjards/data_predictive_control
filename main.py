
# standard inputs
import numpy as np
import json
from scipy.linalg import solve_discrete_are

# custom imports
import mpc 
import visualization.plot as plot
import plant    # simulates actual plant dynamics

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------

# load parameters from config
with open('config.json') as f:
    cfg = json.load(f)

A   = np.array(cfg['A'])
B   = np.array(cfg['B'])
Q   = np.diag(cfg['Q_diag'])
R   = np.diag(cfg['R_diag'])
P   = np.diag(cfg['P_diag'])
x0  = np.array(cfg['x0'], dtype=float)
u0  = np.array(cfg['u0'], dtype=float)
h   = cfg['h']
m   = cfg['m']
constraints = cfg['constraints']

# initialize the MPC controller
controller = mpc.MPC(A, B, Q, R, P, x0, u0, h, m, constraints)

# ------------------------------------------------------------------
# Simulation
# ------------------------------------------------------------------
Ts = 100
x = x0.copy()
u = u0.copy()

# storage
state_history       = [x.copy()]
predicted_sequences = []
input_history       = []

for k in range(Ts):

    # run controller
    controller.solve(x, u)

    # store predicted sequence 
    predicted_sequences.append(controller.result_state_sequence.reshape(h, controller.nx).copy())

    # apply first control input and advance state
    u = controller.result_control_next.flatten()
    input_history.append(u.copy())
    
    x, u = plant.evolve(x, u)

    # store actual state
    state_history.append(x.copy())

    if np.linalg.norm(x) < 1e-3:
        print(f"Converged at step {k + 1}")
        break

print(f"Simulation complete: {len(state_history) - 1} steps, "
      f"final state norm: {np.linalg.norm(x):.4f}")


# ------------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------------
plot.animate_trajectory(state_history, predicted_sequences, solve_discrete_are(A, B, Q, R),
                        filename='visualization/animations/trajectory.gif')
plot.plot_inputs(input_history, constraints, filename='visualization/plots/inputs.png')
plot.plot_velocities(state_history, constraints, filename='visualization/plots/velocities.png')


