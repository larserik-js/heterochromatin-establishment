import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
import os

# Animation packages
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# Simulation class
from simulation_class import Simulation

# Pathname
from parameters import pathname

# External animation file
from animation_class import Animation

# Takes a list of torch tensors, pickles them
def write_pkl(var_list, filename):
    filename = '/home/lars/Documents/masters_thesis/' + filename + '.pkl'

    # Detach tensors and turn them into numpy arrays
    new_var_list = []
    for var in var_list:
        if torch.is_tensor(var):
            new_var_list.append(var.detach().numpy())
        else:
            new_var_list.append(var)

    # Write to pkl
    with open(filename, 'wb') as f:
        pickle.dump(new_var_list, f)

def save_data(sim_obj):
    ## Save final state and statistics
    # Filenames
    parameter_string = sim_obj.params_filename
    fs_filename =  f'statistics/final_state/final_state_' + parameter_string
    interactions_filename =  f'statistics/interactions/interactions_' + parameter_string
    rg_filename = f'statistics/RG/RG_' + parameter_string
    Rs_filename = f'statistics/Rs/Rs_' + parameter_string
    correlation_filename = f'statistics/correlation/correlation_' + parameter_string
    states_filename = f'statistics/states/states_' + parameter_string
    correlation_times_filename = f'statistics/correlation_times/correlation_times_' + parameter_string

    # Final state
    x_final, y_final, z_final = sim_obj.X[:, 0], sim_obj.X[:, 1], sim_obj.X[:, 2]
    pickle_var_list = [x_final, y_final, z_final, sim_obj.states,
                       sim_obj.state_colors, sim_obj.state_names, sim_obj.plot_dim]
    write_pkl(pickle_var_list, fs_filename)

    # Statistics
    pickle_var_list = [sim_obj.N, sim_obj.noise, sim_obj.interaction_idx_difference, sim_obj.average_lifetimes]
    write_pkl(pickle_var_list, interactions_filename)

    # Radius of gyration
    pickle_var_list = [sim_obj.radius_of_gyration]
    write_pkl(pickle_var_list, rg_filename)

    # End-to-end distance
    pickle_var_list = [sim_obj.Rs]
    write_pkl(pickle_var_list, Rs_filename)

    # Correlation
    pickle_var_list = [sim_obj.correlation_sums]
    write_pkl(pickle_var_list, correlation_filename)

    # States
    pickle_var_list = [sim_obj.state_statistics]
    write_pkl(pickle_var_list, states_filename)

    # Correlation times
    pickle_var_list = [sim_obj.correlation_times]
    write_pkl(pickle_var_list, correlation_times_filename)

# Runs the script
# from memory_profiler import profile
# @profile
def run(N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight, alpha_1, alpha_2, beta, stats_t_interval,
        seed, test_mode, animate, allow_state_change, cenH, write_cenH_data, verbose):

    # torch.set_num_threads(1)
    print(f'Started simulation with noise = {noise}')

    # Fix seed value
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create simulation object
    sim_obj = Simulation(N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight, alpha_1, alpha_2, beta,
                         stats_t_interval, seed, allow_state_change, cenH, write_cenH_data)

    # Save initial state for plotting
    x_init = copy.deepcopy(sim_obj.X[:,0])
    y_init = copy.deepcopy(sim_obj.X[:,1])
    z_init = copy.deepcopy(sim_obj.X[:,2])

    coords_init = [x_init, y_init, z_init]

    # Simulation loop
    if animate:
        # Create animation object
        anim_obj = Animation(sim_obj, t_total, coords_init)

        # The animation loop
        # The function 'animation_loop' gets called t_total times
        anim = FuncAnimation(anim_obj.fig_anim, anim_obj.animation_loop, frames=anim_obj.frame_generator,
                             interval=100, save_count=t_total + 10)
        # Format and save
        writergif = animation.PillowWriter(fps=30)

        # Just save a test gif
        if test_mode:
            filename = pathname + 'animations/test.gif'
            anim.save(filename, dpi=200, writer=writergif)

        # Save a named gif, plus pickled final states and statistics
        else:
            ## Save animation
            filename = pathname + 'animations/' + sim_obj.params_filename + '.gif'
            anim.save(filename, dpi=200, writer=writergif)

            ## Save data
            save_data(sim_obj)

    else:
        # Iterate
        for t in range(t_total):
            # Print progress
            if verbose:
                if (t + 1) % (t_total / 10) == 0:
                    print(f'{os.getpid()} : Time-step: {t + 1} / {t_total}   (noise={noise})')

            # Update
            sim_obj.update()

            # Increment no. of time-steps
            sim_obj.t += 1

        # Just plot without saving:
        if test_mode:
            ## Make figure
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # # Plot initial state
            # with torch.no_grad():
            #     sim_obj.plot(x_init, y_init, z_init, ax, label='Initial state', ls='--')

            # Plot final state
            x_final, y_final, z_final = sim_obj.X[:,0], sim_obj.X[:,1], sim_obj.X[:,2]
            #u_final, v_final, w_final = sim_obj.P[:, 0], sim_obj.P[:, 1], sim_obj.P[:, 2]

            with torch.no_grad():
                sim_obj.plot(x_final, y_final, z_final, ax, label='Final state')

            # Plot MD
            sim_obj.finalize_plot(ax)
            #
            # # Plot statistics
            # sim_obj.plot_statistics()
            #import plotter
            # ts = torch.arange(len(sim_obj.state_statistics[0]))
            # plt.plot(ts, sim_obj.state_statistics[0], lw=0.1, label='State S')
            # plt.plot(ts, sim_obj.state_statistics[1], lw=0.1, label='State U')
            # plt.plot(ts, sim_obj.state_statistics[2], lw=0.1, label='State A')
            #plt.legend()
            #plt.show()

        # Just save statistics, no plotting
        else:

            print(sim_obj.t)
            ## Save data
            save_data(sim_obj)

    print(f'Finished simulation with noise = {noise}')

##############################################################################
##############################################################################