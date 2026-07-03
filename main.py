
# standard inputs
from cffi import model
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
with open('config_mpc.json') as f:
    cfg = json.load(f)

Ts = cfg['Ts']  
Tf = cfg['Tf']
A   = np.array(cfg['A'])
B   = np.array(cfg['B'])
Q   = np.diag(cfg['Q_diag'])
R   = np.diag(cfg['R_diag'])
P   = cfg['P_diag']  # may be 'dare' string or a list of diagonal values
x0  = np.array(cfg['x0'], dtype=float)
u0  = np.array(cfg['u0'], dtype=float)
h   = cfg['h']
m   = cfg['m']
constraints = cfg['constraints']
disturbance = cfg['disturbance'] 

# initialize the MPC controller
controller = mpc.MPC(A, B, Q, R, P, x0, u0, h, m, constraints, disturbance)

# initialize plant modeller 
modeller = mpc.Modeller(0*A, 0*B, learning_rate=0.1, window_size=10, optimizer = 'least_squares', update_parameters_rate=20)

# initial state and storage
x = x0.copy()
u = u0.copy()
state_history       = [x.copy()]
predicted_sequences = []
input_history       = []

# //DEBUG//: open log file for model estimates
model_log = open('model_log.txt', 'w')
model_log.write('step,A_hat,B_hat\n') # headers

# ------------------------------------------------------------------
# Modelling phase: random step inputs (ZOH) to excite system modes
# before handing off to MPC
# ------------------------------------------------------------------
u_max          = 0.5*np.array(constraints['u_max'])
u_min          = 0.5*np.array(constraints['u_min'])
excite_hold    = 0.2                                    # seconds to hold each random input
excite_hold_steps = int(excite_hold / Ts)               # steps per hold
excite_max_steps  = int(120 / Ts)                       # max excitation duration (seconds)

print(f"Modelling phase: up to {excite_max_steps} steps ({excite_hold_steps} steps/hold)")

rng = np.random.default_rng(seed=42)
found_stable_model = False
k = 0
u_exc = rng.uniform(u_min, u_max)                       # initial random input
while k < excite_max_steps:
    # draw a new random input every excite_hold_steps
    if k % excite_hold_steps == 0:
        u_exc = rng.uniform(u_min, u_max)
    modeller.update(x, u_exc)
    x = plant.evolve(x, u_exc)
    state_history.append(x.copy())
    input_history.append(u_exc.copy())
    model_log.write(f'exc_{k + 1},{modeller.A_hat.tolist()},{modeller.B_hat.tolist()}\n')
    if modeller.viable and not modeller.shared:
        found_stable_model = True
        controller.A = modeller.A_hat
        controller.B = modeller.B_hat
        controller.update_internal_parameters()
        modeller.shared = True
    k += 1

model_log.close()
print(f"Modelling complete. Model viable: {modeller.viable}")

# preserve excitation history for visualization
keep_excitation_history = False
if keep_excitation_history:
    excite_state_history = state_history.copy()
    excite_input_history = input_history.copy()
else:
    excite_state_history = []
    excite_input_history = []


# reset state to x0 — excitation was for identification only, MPC starts fresh
x = x0.copy()
state_history = [x.copy()]
input_history = []

# ------------------------------------------------------------------
# Controller phase: run MPC with online model updates
# ------------------------------------------------------------------

for k in range(int(Tf / Ts)):

    # run controller
    controller.solve(x, u)

    # store predicted sequence 
    predicted_sequences.append(controller.result_state_sequence.reshape(h, controller.nx).copy())

    # apply first control input and advance state
    u = controller.result_control_next.flatten()
    input_history.append(u.copy())

    # now done above
    '''
    # update model with current state and input (before evolving)
    modeller.update(x, u)

    # //DEBUG//:
    model_log.write(f'{k},{modeller.A_hat.tolist()},{modeller.B_hat.tolist()}\n')
    '''

    x = plant.evolve(x, u)

    # store actual state
    state_history.append(x.copy())

    # now done above
    '''
    if modeller.viable and not modeller.shared:
        controller.A = modeller.A_hat
        controller.B = modeller.B_hat
        controller.update_internal_parameters()
        modeller.shared = True
    '''


    if np.linalg.norm(x) < 1e-3:
        print(f"Converged at step {k + 1}")
        break

print(f"Simulation complete: {len(state_history) - 1} steps, "
      f"final state distance from goal: {np.linalg.norm(x):.4f}")

#//DEBUG//:
'''
model_log.close()
'''

# ------------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------------
# combine excitation + MPC histories for full-trajectory plots
full_state_history = excite_state_history + state_history[1:]  # avoid duplicate x0
full_input_history = excite_input_history + input_history
# predicted_sequences only covers MPC phase (no predictions during excitation)
excite_predicted = [None] * len(excite_input_history)

plot.animate_trajectory(full_state_history, excite_predicted + predicted_sequences, solve_discrete_are(A, B, Q, R),
                        filename='visualization/animations/trajectory.gif')
plot.plot_inputs(full_input_history, constraints, filename='visualization/plots/inputs.png')
plot.plot_velocities(full_state_history, constraints, filename='visualization/plots/velocities.png')


