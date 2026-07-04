# standard inputs
import numpy as np
import json
from scipy.linalg import solve_discrete_are

# custom imports
import plant as le_plant 
import mpc 
import visualization.plot as plot
  
# ------------------------------------------------------------------
# Model dynamics by exciting plant modes
# ------------------------------------------------------------------

# initialize the plant
plant = le_plant.Plant()
x = plant.x0.copy()

# initialize modeller 
modeller = mpc.Modeller()

print(f"Exciting the plant modes for modelling...")
x, excite_state_history, excite_input_history = modeller.excite(plant, x, [], [])
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

print(f"Simulation complete: {len(state_history) - 1} steps")
print(f"Final state distance from goal: {np.linalg.norm(x):.4f}")


# ------------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------------
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

animate_path = cfg['animate_path']
plot_inputs_path = cfg['plot_inputs_path']
plot_velocities_path = cfg['plot_velocities_path']

plot.animate_trajectory(full_state_history, excite_predicted + predicted_sequences, solve_discrete_are(controller.A, controller.B, controller.Q, controller.R),filename=animate_path)
plot.plot_inputs(full_input_history, controller.constraints, filename=plot_inputs_path)
plot.plot_velocities(full_state_history, controller.constraints, filename=plot_velocities_path)


