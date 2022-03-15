import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from numba import njit
import time

# Animation packages
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# Simulation class
from simulation_class import Simulation

# Pathname
from formatting import get_project_folder, create_param_string

# External animation file
from animation_class import Animation, create_animation_directory

# Takes a list of torch tensors, pickles them
def write_pkl(var_list, pathname, filename):
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

def save_data(sim_obj, pathname):
    ## Save final state and statistics
    # Filenames
    parameter_string = sim_obj.params_filename

    fs_filename = 'final_state/final_state_' + parameter_string
    interactions_filename = 'interactions/interactions_' + parameter_string
    Rs_filename = 'Rs/Rs_' + parameter_string
    dist_vecs_to_com_filename = 'dist_vecs_to_com/dist_vecs_to_com_' + parameter_string
    correlation_filename = 'correlation/correlation_' + parameter_string
    states_filename = 'states/states_' + parameter_string
    states_time_space_filename = 'states_time_space/states_time_space_' + parameter_string
    correlation_times_filename = 'correlation_times/correlation_times_' + parameter_string
    successful_conversions_filename = 'successful_conversions/successful_conversions_' + parameter_string

    # Final state
    x_final, y_final, z_final = sim_obj.X[:, 0], sim_obj.X[:, 1], sim_obj.X[:, 2]
    pickle_var_list = [x_final, y_final, z_final, sim_obj.states]
    write_pkl(pickle_var_list, pathname, fs_filename)

    # # Save active polymer with pressure before dynamics
    # write_filename = pathname + 'quasi_random_initial_states_pressure_before_dynamics/' \
    #                 + f'pressure={sim_obj.U_pressure_weight:.2f}/seed={sim_obj.seed}.pkl'
    # # Write to pkl
    # with open(write_filename, 'wb') as f:
    #     pickle.dump(sim_obj.X, f)


    # Interactions and lifetimes
    pickle_var_list = [sim_obj.N, sim_obj.noise, sim_obj.interaction_idx_difference, sim_obj.average_lifetimes]
    write_pkl(pickle_var_list, pathname, interactions_filename)

    # End-to-end distance
    pickle_var_list = [sim_obj.Rs]
    write_pkl(pickle_var_list, pathname, Rs_filename)

    # Distance vectors to center of mass
    pickle_var_list = [sim_obj.dist_vecs_to_com]
    write_pkl(pickle_var_list, pathname, dist_vecs_to_com_filename)

    # Correlation
    pickle_var_list = [sim_obj.correlation_sums]
    write_pkl(pickle_var_list, pathname, correlation_filename)

    # States
    pickle_var_list = [sim_obj.state_statistics]
    write_pkl(pickle_var_list, pathname, states_filename)

    # States in time and space
    pickle_var_list = [sim_obj.states_time_space]
    write_pkl(pickle_var_list, pathname, states_time_space_filename)

    # Correlation times
    pickle_var_list = [sim_obj.correlation_times]
    write_pkl(pickle_var_list, pathname, correlation_times_filename)

    # Succesful conversions
    pickle_var_list = [sim_obj.successful_recruited_conversions, sim_obj.successful_noisy_conversions]
    write_pkl(pickle_var_list, pathname, successful_conversions_filename)

# Fix seed value for Numba
@njit
def set_numba_seed(seed):
    np.random.seed(seed)
    return None

# Runs the script
# from memory_profiler import profile
# @profile
def run(run_on_cell, N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight, alpha_1, alpha_2,
        beta, stats_t_interval, set_seed, seed, test_mode, animate, allow_state_change, initial_state, cell_division,
        cenH_size, cenH_init_idx, write_cenH_data, barriers):

    # Number of failed simulation attempts
    n_failed_simulations = 0

    while True:
        # Runs simulation
        try:
            # torch.set_num_threads(1)
            print(f'Started simulation with seed = {seed}.')

            # Project folder
            pathname = get_project_folder(run_on_cell)

            # Set seed values
            if set_seed:
                np.random.seed(seed)
                torch.manual_seed(seed)
                set_numba_seed(seed)

            # Create simulation object
            sim_obj = Simulation(pathname, N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight, alpha_1,
                                 alpha_2, beta, stats_t_interval, seed, allow_state_change, initial_state, cell_division,
                                 cenH_size, cenH_init_idx, write_cenH_data, barriers)

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
                N_IMAGES = 500
                iterations_per_image = int(t_total / N_IMAGES)

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

                        # Set dpi=60 to get < 50MB data
                        sim_obj.fig.savefig(animation_folder + f'{image_idx:03d}', dpi=100)

                # Save data
                save_data(sim_obj, pathname)

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
                            return sim_obj.t

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
                    save_data(sim_obj, pathname)

            print(f'Finished simulation with seed = {seed}.')

        except Exception as e:
            n_failed_simulations += 1
            message = f'Simulation failed: {e}. Restarting in 10 s.'
            print(message)
            time.sleep(10)

        # If no exception occurred, break the 'while True' loop
        else:
            break

        # Limits the total number of failed simulations
        finally:
            if n_failed_simulations >= 100:
                break

    return None

##############################################################################
##############################################################################