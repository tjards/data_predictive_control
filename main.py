# standard inputs
import numpy as np
import json
from scipy.linalg import solve_discrete_are

# custom imports
import plant as le_plant 
import mpc 
import visualization.plot as plot
from data_manager import Dataset
  
# initial dataset
#data = Dataset()

# ------------------------------------------------------------------
# Model dynamics by exciting plant modes
# ------------------------------------------------------------------

# initialize the plant
plant = le_plant.Plant()
x = plant.x0.copy()

# initialize modeller 
modeller = mpc.Modeller()

print(f"Exciting the plant modes for modelling...")

x, excite_state_history, excite_input_history, A_hat_history, B_hat_history, step_history = modeller.excite(plant, x)

annotated_step_history = [f"excite_{s}" for s in step_history]
#data.load(step = annotated_step_history, A_hat = A_hat_history, B_hat = B_hat_history, state = excite_state_history, input = excite_input_history)
#data.store(flush_after=True)

print(f"Modelling complete. Model viable: {modeller.viable}")

# ------------------------------------------------------------------
# Run the Controller 
# ------------------------------------------------------------------

# note: add a feasibility check out to necessary horizon

# initialize the MPC controller
controller = mpc.MPC(x)

# load the model parameters into the controller
controller.A = modeller.A_hat
controller.B = modeller.B_hat
controller.new_model_parameters = True

# check feasibility of the current state and input
controller.confirm_feasibility(x, controller.u0)

# initialize storage 
state_history = [x.copy()]
input_history = []
predicted_sequences = []

# initialize the control input
u = controller.u0

for k in range(int(controller.Tf / controller.Ts)):

    # run controller
    controller.solve(x, u)

    # store predicted sequence 
    predicted_sequences.append(controller.result_state_sequence.reshape(controller.h, controller.nx).copy())

    # apply first control input and advance state
    u = controller.result_control_next.flatten()
    input_history.append(u.copy())

    x = plant.evolve(x, u)

    # store actual state
    state_history.append(x.copy())

    #data.stage(step = k, A_hat = controller.A, B_hat = controller.B, d_hat = controller.d_hat, d = plant.d, target = None, state = x, input = u, plan = predicted_sequences)
    #data.store(flush_after=True)


'''
print(f"Simulation complete: {len(state_history) - 1} steps")
print(f"Final state distance from goal: {np.linalg.norm(x):.4f}")

#data_loaded = data.load_dataset_log("data/dataset.h5")

print('excite_input_history shape: ', excite_input_history[0].shape)
print('length: ', len(excite_input_history)) 

print('k: ', k)
print('type: ', type(k)) 

print('controller.A: ', controller.A.shape)
print('length: ', len(controller.A)) 

print('controller.B: ', controller.B.shape)
print('length: ', len(controller.B))

print('plant.d: ', plant.d.shape)
print('length: ', len(plant.d))

print('controller.d_hat: ', controller.d_hat.shape)
print('length: ', len(controller.d_hat))

print('input_history: ', input_history[0].shape)
print('length: ', len(input_history))

print('excite_state_history: ', excite_state_history[0].shape)
print('length: ', len(excite_state_history))

print('state_history: ', state_history[0].shape)
print('length: ', len(state_history))   

print('x: ', x.shape)
print('u: ', u.shape)
print('predicted_sequences: ', predicted_sequences[0].shape)
print('length: ', len(predicted_sequences))
'''


# k
# controller.A
# contoller.B
# plant.d
# controller.d_hat
# input_history
# excite_state_history 
# state_history
# x
# u
# predicted_sequences 



# ------------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------------

with open('configs/config_visualization.json') as f:
    cfg = json.load(f)

animate_path            = cfg['animate_path']
plot_inputs_path        = cfg['plot_inputs_path']
plot_velocities_path    = cfg['plot_velocities_path']


'''
keep_excitation_history = True

if not keep_excitation_history:
    plot_start = data_loaded['step'].index(0)
else:
    plot_start = 0

sliced_states = np.vstack(data_loaded['state'][plot_start:])
sliced_inputs = np.vstack(data_loaded['input'][plot_start:])
sliced_plan = data_loaded['plan'][plot_start:]



plot.animate_trajectory(sliced_states, sliced_plan, solve_discrete_are(controller.A, controller.B, controller.Q, controller.R),filename=animate_path)
plot.plot_inputs(sliced_inputs, controller.constraints, filename=plot_inputs_path)
plot.plot_velocities(sliced_states, controller.constraints, filename=plot_velocities_path)
'''

# ---
keep_excitation_history = True

if keep_excitation_history:
    full_state_history = excite_state_history + state_history[1:]  # avoid duplicate x0
    full_input_history = excite_input_history + input_history
else:
    full_state_history = state_history
    full_input_history = input_history

excite_predicted = [None] * len(excite_input_history)

with open('configs/config_visualization.json') as f:
    cfg = json.load(f)

animate_path            = cfg['animate_path']
plot_inputs_path        = cfg['plot_inputs_path']
plot_velocities_path    = cfg['plot_velocities_path']

plot.animate_trajectory(full_state_history, excite_predicted + predicted_sequences, solve_discrete_are(controller.A, controller.B, controller.Q, controller.R),filename=animate_path)
plot.plot_inputs(full_input_history, controller.constraints, filename=plot_inputs_path)
plot.plot_velocities(full_state_history, controller.constraints, filename=plot_velocities_path)


