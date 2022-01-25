import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
import os
from numba import njit

# Animation packages
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# Simulation class
from simulation_class import Simulation

# Pathname
from formatting import pathname, create_param_string

# External animation file
from animation_class import Animation, create_animation_directory

# Takes a list of torch tensors, pickles them
def write_pkl(var_list, filename):
    filename = pathname + 'data/statistics/' + filename + '.pkl'

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

    fs_filename =  'final_state/final_state_' + parameter_string
    interactions_filename =  'interactions/interactions_' + parameter_string
    Rs_filename = 'Rs/Rs_' + parameter_string
    dist_vecs_to_com_filename = 'dist_vecs_to_com/dist_vecs_to_com_' + parameter_string
    correlation_filename = 'correlation/correlation_' + parameter_string
    states_filename = 'states/states_' + parameter_string
    states_time_space_filename = 'states_time_space/states_time_space_' + parameter_string
    correlation_times_filename = 'correlation_times/correlation_times_' + parameter_string
    succesful_recruited_conversions_filename = 'succesful_recruited_conversions/succesful_recruited_conversions_' + parameter_string

    # Final state
    x_final, y_final, z_final = sim_obj.X[:, 0], sim_obj.X[:, 1], sim_obj.X[:, 2]
    pickle_var_list = [x_final, y_final, z_final, sim_obj.states]
    write_pkl(pickle_var_list, fs_filename)

    # Statistics
    pickle_var_list = [sim_obj.N, sim_obj.noise, sim_obj.interaction_idx_difference, sim_obj.average_lifetimes]
    write_pkl(pickle_var_list, interactions_filename)

    # End-to-end distance
    pickle_var_list = [sim_obj.Rs]
    write_pkl(pickle_var_list, Rs_filename)

    # Distance vectors to center of mass
    pickle_var_list = [sim_obj.dist_vecs_to_com]
    write_pkl(pickle_var_list, dist_vecs_to_com_filename)

    # Correlation
    pickle_var_list = [sim_obj.correlation_sums]
    write_pkl(pickle_var_list, correlation_filename)

    # States
    pickle_var_list = [sim_obj.state_statistics]
    write_pkl(pickle_var_list, states_filename)

    # States in time and space
    pickle_var_list = [sim_obj.states_time_space]
    write_pkl(pickle_var_list, states_time_space_filename)

    # Correlation times
    pickle_var_list = [sim_obj.correlation_times]
    write_pkl(pickle_var_list, correlation_times_filename)

    # Succesful recruited conversions
    pickle_var_list = [sim_obj.succesful_recruited_conversions]
    write_pkl(pickle_var_list, succesful_recruited_conversions_filename)

# Fix seed value for Numba
@njit
def set_numba_seed(seed):
    np.random.seed(seed)
    return None

# Runs the script
# from memory_profiler import profile
# @profile
def run(N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight, alpha_1, alpha_2, beta, stats_t_interval,
        seed, test_mode, animate, allow_state_change, initial_state, cell_division, cenH_size, cenH_init_idx,
        write_cenH_data, barriers):

    # torch.set_num_threads(1)
    print(f'Started simulation with seed = {seed}.')

    # Fix seed values
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_numba_seed(seed)

    # Create simulation object
    sim_obj = Simulation(N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight, alpha_1, alpha_2, beta,
                         stats_t_interval, seed, allow_state_change, initial_state, cell_division, cenH_size,
                         cenH_init_idx, write_cenH_data, barriers)

    # Simulation loop
    if animate:
        # Create destination folder for the individual images
        if test_mode:
            animation_folder = pathname + 'data/animations/test/'
            create_animation_directory(animation_folder)
        else:
            param_string = create_param_string(U_pressure_weight, initial_state, cenH_size, cenH_init_idx,
                                               cell_division, barriers, N, t_total, noise, alpha_1, alpha_2, beta, seed)
            animation_folder = pathname + 'data/animations/' + param_string + '/'
            create_animation_directory(animation_folder)

        # Iterate
        # Ensures that a total of 500 images will be created
        n_images = 500
        iterations_per_image = int(t_total / n_images)

        # Filename formatting
        image_idx = 0

        for t in range(t_total):
            # Print progress
            if (t + 1) % (t_total / 10) == 0:
                print(f'{os.getpid()} : Time-step: {t + 1} / {t_total}')

            # Update
            sim_obj.update()

            # Increment no. of time-steps
            sim_obj.t += 1

            # Save figure
            if t%iterations_per_image == 0:
                # Plot
                sim_obj.plot()
                image_idx += 1
                sim_obj.fig.savefig(animation_folder + f'{image_idx:03d}', dpi=100)

        # Save data
        save_data(sim_obj)

    # No animation
    else:
        # Iterate
        for t in range(t_total):
            # Print progress
            if (t + 1) % (t_total / 10) == 0:
                print(f'{os.getpid()} : Time-step: {t + 1} / {t_total}')

            # Update
            sim_obj.update()

            # Increment no. of time-steps
            sim_obj.t += 1

            if not test_mode:
                # if sim_obj.end_to_end_vec_dot <= 0:
                #     data_file = open(f'/home/lars/Documents/masters_thesis/statistics/end_to_end_perpendicular_times_N={N}'
                #                      + f'_t_total={t_total}_noise={noise:.4f}' f'_alpha_1={alpha_1:.5f}_alpha_2={alpha_2:.5f}'
                #                      + f'_beta={beta:.5f}.txt', 'a')
                #     data_file.write(str(sim_obj.t) + ',' + str(sim_obj.seed) + '\n')
                #     data_file.close()
                #     print(f'Wrote to file at seed {sim_obj.seed}')
                #     break
                if sim_obj.stable_silent == True:
                    break

        # Just plot final state without saving
        if test_mode:
            # try:
            #     os.mkdir(pathname + f'quasi_random_initial_states_pressure_before_dynamics/pressure={U_pressure_weight:.2f}')
            # except FileExistsError:
            #     pass
            #
            # filename = pathname + f'/quasi_random_initial_states_pressure_before_dynamics/pressure={U_pressure_weight:.2f}/seed={seed}.pkl'
            # # Detach tensors and turn them into numpy arrays
            # new_var_list = []
            # var_list = [sim_obj.X]
            #
            # for var in var_list:
            #     if torch.is_tensor(var):
            #         new_var_list.append(var.detach().numpy())
            #     else:
            #         new_var_list.append(var)
            #
            # # Write to pkl
            # with open(filename, 'wb') as f:
            #     pickle.dump(new_var_list, f)
            #     print(f'Wrote to {filename}')

            with torch.no_grad():
               sim_obj.plot()
               plt.show()

        # Just save statistics, no plotting
        else:
            # Save data
            save_data(sim_obj)

    print(f'Finished simulation with seed = {seed}.')

##############################################################################
##############################################################################