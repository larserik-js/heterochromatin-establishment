import pickle
import os
import time

import numpy as np
import torch
from numba import njit

from simulation_class import Simulation
from formatting import create_param_string, make_directory


# Takes a list of torch tensors, pickles them
def write_pkl(var_list, output_dir, filename):
    filename = output_dir + 'statistics/' + filename + '.pkl'

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


def save_data(sim_obj, output_dir):
    parameter_string = sim_obj.params_filename

    # Keys: directory name in which to save files
    # Values: list of variables to pickle and save in aforementioned directories
    directories_and_variables = {
        # Correlation
        'correlation': [sim_obj.correlation_sums],

        # Correlation times
        'correlation_times': [sim_obj.correlation_times],

        # Distance vectors to center of mass
        'dist_vecs_to_com': [sim_obj.dist_vecs_to_com],

        # Final state
        'final_state': [sim_obj.X[:, 0], sim_obj.X[:, 1], sim_obj.X[:, 2], sim_obj.states],

        # Interactions and lifetimes
        'interactions': [sim_obj.N, sim_obj.noise, sim_obj.interaction_idx_difference, sim_obj.average_lifetimes],

        # End-to-end distance
        'Rs': [sim_obj.Rs],

        # States
        'states': [sim_obj.state_statistics],

        # States in time and space
        'states_time_space': [sim_obj.states_time_space],

        # Succesful conversions
        'successful_conversions': [sim_obj.successful_recruited_conversions, sim_obj.successful_noisy_conversions]
    }

    # Pickle and save data
    for dir_name, var_list in directories_and_variables.items():
        write_pkl(var_list, output_dir, dir_name + '/' + parameter_string)


# Fix seed value for Numba
@njit
def set_numba_seed(seed):
    np.random.seed(seed)


# Runs the script
# from memory_profiler import profile
# @profile
def run(model, project_dir, input_dir, output_dir, N, l0, noise, dt, t_total, U_two_interaction_weight, rms, alpha_1,
        alpha_2, beta, set_seed, seed, animate, allow_state_change, initial_state, cell_division, cenH_size,
        cenH_init_idx, write_cenH_data, ATF1_idx):

    # Number of failed simulation attempts
    n_failed_simulations = 0

    while n_failed_simulations <= 10000:
        # Runs simulation
        try:
            # torch.set_num_threads(1)
            print(f'Started simulation with seed = {seed}.')

            # Set seed values
            if set_seed:
                np.random.seed(seed)
                torch.manual_seed(seed)
                set_numba_seed(seed)

            # Create simulation object
            sim_obj = Simulation(model, project_dir, input_dir, output_dir, N, l0, noise, dt, t_total,
                                 U_two_interaction_weight, rms, alpha_1, alpha_2, beta, seed, allow_state_change,
                                 initial_state, cell_division, cenH_size, cenH_init_idx, write_cenH_data, ATF1_idx)

            # Folder for saving animation images
            param_string = create_param_string(model, rms, initial_state, cenH_size, cenH_init_idx,
                                               ATF1_idx, cell_division, N, t_total, noise, alpha_1, alpha_2, beta, seed)

            animation_folder = output_dir + '/animations/' + param_string + '/'

            # Ensures that a total of 500 images will be created
            N_IMAGES = 500
            iterations_per_image = int(t_total / N_IMAGES)

            # Index for image filename
            image_idx = 0

            # Make folder for the individual images
            if animate:
                make_directory(animation_folder)
            else:
                pass

            # Simulation loop
            for t in range(t_total):
                # Print progress
                if (t + 1) % (t_total / 10) == 0:
                    print(f'{os.getpid()} : Time-step: {t + 1} / {t_total}')

                # Update
                sim_obj.update()

                # Increment no. of time-steps
                sim_obj.t += 1

                # Save image for animation
                if animate:
                    # Save figure
                    if t % iterations_per_image == 0:
                        # Plot
                        sim_obj.plot()
                        image_idx += 1

                        # Set dpi=60 to get < 50MB data
                        sim_obj.fig.savefig(animation_folder + f'{image_idx:03d}', dpi=100)

                    # No image saved
                    else:
                        pass

                # Write cenH data only if no animation
                else:
                    # Ends simulation if > 90% of the polymer is silent
                    if write_cenH_data and sim_obj.stable_silent:
                        return sim_obj.t

                    # If not, continue simulation
                    else:
                        pass

            # Save data
            save_data(sim_obj, output_dir)

            print(f'Finished simulation with seed = {seed}.')

        except Exception as e:
            n_failed_simulations += 1
            message = f'Simulation failed: {e}. Restarting in 10 s.'
            print(message)
            time.sleep(10)

        # If no exception occurred, break the while-loop
        else:
            break

    return None