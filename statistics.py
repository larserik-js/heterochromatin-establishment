import torch
import numpy as np
from scipy import ndimage
from copy import deepcopy
from numba import njit
import matplotlib.pyplot as plt
from formatting import edit_stable_silent_times_file
from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from simulation_class import Simulation

# Calculate radius of gyration
def update_dist_vecs_to_com(sim_obj):
    t = sim_obj.t
    interval = int(sim_obj.t_total / sim_obj.dist_vecs_to_com.shape[0])

    if t%interval == 0:
        t_idx = int(t/interval)
        # The distance from each individual nucleosome to the center of mass
        sim_obj.dist_vecs_to_com[t_idx] = sim_obj.X - sim_obj.center_of_mass

    return None

# Update the end-to-end vector
def update_Rs(sim_obj):
    t = sim_obj.t
    interval = int(sim_obj.t_total / len(sim_obj.Rs))
    if t%interval == 0:
        t_interval = int(t / interval)
        sim_obj.Rs[t_interval] = torch.linalg.norm(sim_obj.X[-1] - sim_obj.X[0], dim=0)

    return None

def end_to_end_dot(sim_obj):
    sim_obj.end_to_end_vec_dot = torch.sum((sim_obj.X[-1] - sim_obj.X[0]) * sim_obj.end_to_end_vec_init)
    return None

def get_correlations(interaction_indices_i, interaction_indices_j, shifts):
    correlation_sums = np.zeros_like(shifts)

    # Loop over shifts
    for s in range(len(shifts)):
        shift = shifts[s]
        sum = 0

        # Loop over interactions
        for k in range(len(interaction_indices_i)):
            # Indices of the interaction pair
            i, j = interaction_indices_i[k], interaction_indices_j[k]

            # Loop over other interactions
            for l in range(len(interaction_indices_i)):
                # The other interaction
                other_interaction_i, other_interaction_j = interaction_indices_i[l], interaction_indices_j[l]

                # Check for any of the two possible zig-zag patterns
                # Same direction
                if other_interaction_i == i + shift and other_interaction_j == j + shift:
                    sum += 1
                elif other_interaction_i == i - shift and other_interaction_j == j - shift:
                    sum += 1

                # Opposite direction
                elif other_interaction_i == i + shift and other_interaction_j == j - shift:
                    sum += 1
                elif other_interaction_i == i - shift and other_interaction_j == j + shift:
                    sum += 1

        correlation_sums[s] = sum

    return correlation_sums

def update_correlation_sums(sim_obj):
    normalized_correlations = get_correlations(sim_obj.interaction_indices_i.numpy(),
                                                    sim_obj.interaction_indices_j.numpy(),
                                                    sim_obj.shifts) / sim_obj.t_half

    sim_obj.correlation_sums += normalized_correlations
    return None

def update_interaction_stats(sim_obj):
    # Interaction only applies to distances lower than l_interacting
    # Relevant interactions are only counted once
    interaction_condition = (sim_obj.interaction_mask_two) & (sim_obj.norms_all < sim_obj.l_interacting) & sim_obj.mask_upper
    interaction_indices_i = torch.where(interaction_condition)[0]
    interaction_indices_j = torch.where(interaction_condition)[1]

    # Interaction index differences
    interaction_distances = torch.abs((interaction_indices_j - interaction_indices_i))
    sim_obj.interaction_idx_difference += torch.bincount(interaction_distances, minlength=sim_obj.N)

    # Average lifetimes
    # If two nucleosomes are (still) interacting, add 1 to the running lifetimes
    sim_obj.running_lifetimes[interaction_indices_i, interaction_indices_j] += 1

    # If two nucleosomes are no longer interacting, reset the running lifetime, and count the reset
    reset_condition = (sim_obj.previous_interaction_mask & torch.logical_not(interaction_condition) & sim_obj.mask_upper)
    reset_indices_i, reset_indices_j = torch.where(reset_condition)[0], torch.where(reset_condition)[1]

    sim_obj.lifetimes[reset_indices_i, reset_indices_j] += sim_obj.running_lifetimes[reset_indices_i, reset_indices_j]
    sim_obj.running_lifetimes[reset_indices_i, reset_indices_j] = 0
    sim_obj.completed_lifetimes[reset_indices_i, reset_indices_j] += 1

    # Finalize statistics
    # Normalizes the lifetimes to average values
    if sim_obj.t == sim_obj.t_total - 1:
        for k in range(len(sim_obj.triu_indices[0])):
            i, j = sim_obj.triu_indices[0][k], sim_obj.triu_indices[1][k]
            idx = j - i
            sim_obj.average_lifetimes[idx] += sim_obj.lifetimes[i, j] / (sim_obj.completed_lifetimes[i, j] + 1e-7)
    return None

def update_states(sim_obj):
    t = sim_obj.t
    interval = int(sim_obj.t_total / sim_obj.state_statistics.shape[1])

    if t%interval == 0:
        t_interval = int(t / interval)
        sim_obj.state_statistics[0, t_interval] = (sim_obj.states == 0).sum()
        sim_obj.state_statistics[1, t_interval] = (sim_obj.states == 1).sum()
        sim_obj.state_statistics[2, t_interval] = (sim_obj.states == 2).sum()

    return None

def update_states_time_space(sim_obj):
    t = sim_obj.t
    interval = int(sim_obj.t_total / sim_obj.states_time_space.shape[0])

    if t%interval == 0:
        t_interval = int(t / interval)
        sim_obj.states_time_space[t_interval] = sim_obj.states

    return None

# For each nucleosome, measures the time it takes for the distance vector to the center of mass to
# rotate more than 90 degrees
def update_correlation_times(sim_obj):
    # Current distance vectors from the nucleosomes to the center of mass
    distance_vecs_to_com = sim_obj.center_of_mass - sim_obj.X

    # The dot product of the initial and current distance vectors from each nucleosome to the center of mass
    dot_products = torch.sum(sim_obj.init_dist_vecs_to_com * distance_vecs_to_com, dim=1)

    # Adds the current time the first time the dot product goes below 0
    sim_obj.correlation_times[(sim_obj.correlation_times == 0) & (dot_products <= 0)] = sim_obj.t / sim_obj.t_total

    return None

def _gather_statistics(sim_obj):
    # Write cenH data
    if sim_obj.write_cenH_data:
        # When half the polymer reaches the silent state
        if torch.sum(sim_obj.states == 0) >= 0.5*sim_obj.N and sim_obj.half_silent_time is None:
            sim_obj.half_silent_time = deepcopy(sim_obj.t)
            _, sim_obj.n_silent_patches = ndimage.label(sim_obj.states == 0)

        # If the polymer collapses to silent, write the time at which this occurs to a .txt file
        if sim_obj.stable_silent == False:
            if torch.sum(sim_obj.states == 0) >= 0.9*sim_obj.N:
                line_str = f'{sim_obj.t},{sim_obj.half_silent_time},{sim_obj.n_silent_patches},{sim_obj.seed}'

                edit_stable_silent_times_file(sim_obj.U_pressure_weight, sim_obj.initial_state, sim_obj.cenH_size,
                                              sim_obj.cenH_init_idx, sim_obj.cell_division, sim_obj.barriers, sim_obj.N,
                                              sim_obj.t_total, sim_obj.noise, sim_obj.alpha_1, sim_obj.alpha_2,
                                              sim_obj.beta, sim_obj.seed, line_str)

                print(f'Wrote to file at seed {sim_obj.seed}')

                # Set the system state to silent. This will trigger the simulation to finish in 'run.py'
                sim_obj.stable_silent = True

            # If the simulation finishes without a polymer collapse, write 'NaN' to the .txt file
            elif sim_obj.t == sim_obj.t_total - 1:
                line_str = f'NaN,NaN,NaN,{sim_obj.seed}'

                edit_stable_silent_times_file(sim_obj.U_pressure_weight, sim_obj.initial_state, sim_obj.cenH_size,
                                              sim_obj.cenH_init_idx,
                                              sim_obj.cell_division, sim_obj.barriers, sim_obj.N, sim_obj.t_total,
                                              sim_obj.noise, sim_obj.alpha_1, sim_obj.alpha_2, sim_obj.beta,
                                              sim_obj.seed, line_str)

                print(f'Wrote to file at seed {sim_obj.seed}')

            # Continue the simulation
            else:
                pass

        # If stable_silent == True, the simulation will exit before this step, through 'run.py'
        else:
            pass


    # Do not gather any statistics if cenH data is to be written
    # This is because the different simulations will have different length,
    # and the data will be hard to compare
    else:
        # Calculate the dot product of the end-to-end vector with the initial end-to-end vector
        #end_to_end_dot(sim_obj)

        # Update R
        #update_Rs(sim_obj)

        # Count number of particles in each state
        #update_states(sim_obj)
        update_states_time_space(sim_obj)

        # Update time correlation of polymer
        #update_correlation_times(sim_obj)

        #update_interaction_stats(sim_obj)

        # Update radius of gyration
        update_dist_vecs_to_com(sim_obj)

        # These statistics are taken from halfway through the simulation
        #if sim_obj.t >= sim_obj.t_half:

        #
        #     # # Count correlations by shift
        #     # update_correlation_sums(sim_obj)

    return None