# standard inputs
import numpy as np
import json
from scipy.linalg import solve_discrete_are

# custom imports
import plant as le_plant 
import disturbance_generator 
import mpc 
import visualization.plot as plot
from data_manager import Dataset

# ------------------------------------------------------------------
# Pipeline Setup
# ------------------------------------------------------------------ 
pipeline = {
    'model':    True,
    'control':  True,
    'visuals':  True
}
# ------------------------------------------------------------------
# Initialize plant and data
# ------------------------------------------------------------------

# initialize the global timer
t = 0.0

# initialize plant
plant = le_plant.Plant()
x = plant.x0.copy()

# initialize disturbances
disturbor = disturbance_generator.Disturbance()
d = disturbor.evolve(t)

# initial dataset
with open('configs/config_data.json') as f:
    cfg_dat = json.load(f)
data = Dataset(filepath=cfg_dat["filepath"], overwrite=cfg_dat["overwrite"])

# ------------------------------------------------------------------
# Model dynamics by exciting plant modes (no disturbances)
# ------------------------------------------------------------------
if pipeline['model']:

    # initialize modeller 
    modeller = mpc.Modeller()

    print(f'Modelling started at time: {round(t)} seconds')
    print(f"Exciting the plant modes for modelling...")

    x, t, state_history, input_history, A_hat_history, B_hat_history, step_history = modeller.excite(plant, disturbor, x, t)

    print(f'Modelling completed at time: {round(t)} seconds')

    data.stage(phase = 'modelling', 
            step = step_history, 
            A_hat = A_hat_history, 
            B_hat = B_hat_history, 
            state = state_history, 
            input = input_history,)
    data.store()

    print(f"Modelling complete. Model viable: {modeller.viable}")

    # pull new data
    modelling_data          = data.read('modelling') 

else:

    # pull data from defaults
    data_defaults = Dataset(filepath=cfg_dat["defaults"], overwrite=False)
    modelling_data          = data_defaults.read('modelling')

# extract what we need later 
epsilon = 1e-5
A_hat = modelling_data['A_hat'][-1]
A_hat[A_hat < epsilon] = 0.0
B_hat = modelling_data['B_hat'][-1]
B_hat[B_hat < epsilon] = 0.0

# ------------------------------------------------------------------
# Run the Controller 
# ------------------------------------------------------------------
if pipeline['control']:

    # initialize the MPC controller and load params f
    controller = mpc.MPC(x)

    if controller.use_learned_model:
        controller.A = A_hat    #modeller.A_hat
        controller.B = B_hat    #modeller.B_hat
        controller.new_model_parameters = True
    else:
        print('using first-principles model')

    # check feasibility of the current state and input
    controller.confirm_feasibility(x, controller.u0)

    # initialize the control input
    u = controller.u0

    print(f'Controller started at time: {round(t)} seconds')

    for k in range(int(controller.Tf / controller.Ts)):

        # run controller
        controller.solve(x, u)

        # store predicted sequence 
        current_plan = controller.result_state_sequence.reshape(controller.h,controller.nx,).copy()

        # apply first control input 
        u = controller.result_control_next.flatten()

        # evolve the disturbance
        d = disturbor.evolve(k*controller.Ts)

        # evolve the plant
        x = plant.evolve(x, u, d, disturb=True)

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

        # increment the global timer
        t += controller.Ts

    print(f'Controller completed at time: {round(t)} seconds')
    print(f"Final state distance from goal: {np.linalg.norm(x):.4f}")

    controller_data         = data.read('controller')

else:

    data_defaults = Dataset(filepath=cfg_dat["defaults"], overwrite=False)
    controller_data         = data_defaults.read('controller')

# ------------------------------------------------------------------
# Visualizations
# ------------------------------------------------------------------
if pipeline['visuals']:

    # pull visualization configs 
    with open('configs/config_visualization.json') as f:
        cfg_viz = json.load(f)

    animate_path            = cfg_viz['animate_path']
    plot_inputs_path        = cfg_viz['plot_inputs_path']
    plot_velocities_path    = cfg_viz['plot_velocities_path']
    keep_modelling_history  = cfg_viz['keep_modelling_history']

    with open('configs/config_mpc.json') as f:
        cfg_mpc = json.load(f)
    constraints = cfg_mpc['constraints']

    if keep_modelling_history:
        full_state_history      = list(modelling_data["state"]) + list(controller_data["state"])[1:]  # avoid duplicate x0
        full_input_history      = list(modelling_data["input"]) + list(controller_data["input"])
        predicted_sequences     = [None] * len(modelling_data["state"]) + list(controller_data["plan"])
    else:
        full_state_history      = list(controller_data["state"]) 
        full_input_history      = list(controller_data["input"])
        predicted_sequences     = list(controller_data["plan"])

    #plot.animate_trajectory(full_state_history, predicted_sequences, solve_discrete_are(controller.A, controller.B, controller.Q, controller.R),filename=animate_path)
    
    print('Producing animation...')
    plot.animate_trajectory(full_state_history, predicted_sequences, V = None, filename=animate_path)
    print('Producing plots...')
    plot.plot_inputs(full_input_history, constraints, filename=plot_inputs_path)
    plot.plot_velocities(full_state_history, constraints, filename=plot_velocities_path)


