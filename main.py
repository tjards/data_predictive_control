# standard inputs
import numpy as np
import json
from scipy.linalg import solve_discrete_are

# custom imports
import plant as le_plant 
import mpc 
import visualization.plot as plot
from data_manager import Dataset

# ------------------------------------------------------------------
# Initialize plant and data
# ------------------------------------------------------------------

# initialize plant
plant = le_plant.Plant()
x = plant.x0.copy()

# initial dataset
with open('configs/config_data.json') as f:
    cfg_dat = json.load(f)
data = Dataset(filepath=cfg_dat["filepath"], overwrite=cfg_dat["overwrite"])

# ------------------------------------------------------------------
# Model dynamics by exciting plant modes
# ------------------------------------------------------------------

# initialize modeller 
modeller = mpc.Modeller()

print(f"Exciting the plant modes for modelling...")

x, state_history, input_history, A_hat_history, B_hat_history, step_history = modeller.excite(plant, x)

data.stage(phase = 'modelling', 
           step = step_history, 
           A_hat = A_hat_history, 
           B_hat = B_hat_history, 
           state = state_history, 
           input = input_history,)
data.store()

print(f"Modelling complete. Model viable: {modeller.viable}")

# ------------------------------------------------------------------
# Run the Controller 
# ------------------------------------------------------------------

# note: add a feasibility check out to necessary horizon

# initialize the MPC controller and load params f
controller = mpc.MPC(x)
controller.A = modeller.A_hat
controller.B = modeller.B_hat
controller.new_model_parameters = True

# check feasibility of the current state and input
controller.confirm_feasibility(x, controller.u0)

# initialize the control input
u = controller.u0

for k in range(int(controller.Tf / controller.Ts)):

    # run controller
    controller.solve(x, u)

    # store predicted sequence 
    current_plan = controller.result_state_sequence.reshape(controller.h,controller.nx,).copy()

    # apply first control input and advance state
    u = controller.result_control_next.flatten()
    x = plant.evolve(x, u)

    data.stage(phase = 'controller', 
               step = k, 
               A_hat = controller.A, 
               B_hat = controller.B, 
               d_hat = controller.d_hat, 
               d = plant.d, 
               target = None, 
               state = x, 
               input = u, 
               plan = current_plan)
    data.store(flush_after=True)

print(f"Simulation complete: {int(controller.Tf / controller.Ts)} steps")
print(f"Final state distance from goal: {np.linalg.norm(x):.4f}")

# ------------------------------------------------------------------
# Extract data 
# ------------------------------------------------------------------

modelling_data          = data.read('modelling')
controller_data         = data.read('controller')

# ------------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------------

# pull visualization configs 
with open('configs/config_visualization.json') as f:
    cfg_viz = json.load(f)

animate_path            = cfg_viz['animate_path']
plot_inputs_path        = cfg_viz['plot_inputs_path']
plot_velocities_path    = cfg_viz['plot_velocities_path']
keep_modelling_history  = cfg_viz['keep_modelling_history']

if keep_modelling_history:
    full_state_history      = list(modelling_data["state"]) + list(controller_data["state"])[1:]  # avoid duplicate x0
    full_input_history      = list(modelling_data["input"]) + list(controller_data["input"])
    predicted_sequences     = [None] * len(modelling_data["state"]) + list(controller_data["plan"])
else:
    full_state_history      = list(controller_data["state"]) 
    full_input_history      = list(controller_data["input"])
    predicted_sequences     = list(controller_data["plan"])

plot.animate_trajectory(full_state_history, predicted_sequences, solve_discrete_are(controller.A, controller.B, controller.Q, controller.R),filename=animate_path)
plot.plot_inputs(full_input_history, controller.constraints, filename=plot_inputs_path)
plot.plot_velocities(full_state_history, controller.constraints, filename=plot_velocities_path)


