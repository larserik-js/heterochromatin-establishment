import torch
import numpy as np
from numba import njit
import matplotlib.pyplot as plt

# Calculate radius of gyration
def update_rg(sim_obj):
    distances_to_com = torch.norm(sim_obj.X - sim_obj.center_of_mass)
    ## Add new value of RG to itsim_obj
    # At the end divided by t_half for time average
    sim_obj.radius_of_gyration += torch.sqrt(torch.mean(distances_to_com ** 2))

    # Calculate the average in the last time-step
    if sim_obj.t == sim_obj.t_total - 1:
        sim_obj.radius_of_gyration = sim_obj.radius_of_gyration / sim_obj.t_half
    return None

@njit
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


def update_interaction_idx_differences(sim_obj):
    interaction_distances = torch.abs((sim_obj.interaction_indices_j - sim_obj.interaction_indices_i))
    sim_obj.interaction_idx_difference += torch.bincount(interaction_distances, minlength=sim_obj.N)
    return None


def update_correlation_sums(sim_obj):
    normalized_correlations = get_correlations(sim_obj.interaction_indices_i.numpy(),
                                                    sim_obj.interaction_indices_j.numpy(),
                                                    sim_obj.shifts) / sim_obj.t_half

    sim_obj.correlation_sums += normalized_correlations
    return None


def update_average_lifetimes(sim_obj):
    # If two nucleosomes are (still) interacting, add 1 to the running lifetimes
    sim_obj.running_lifetimes[sim_obj.interaction_indices_i, sim_obj.interaction_indices_j] += 1

    # If two nucleosomes are no longer interacting, reset the running lifetime, and count the reset
    reset_condition = (sim_obj.previous_interaction_mask & torch.logical_not(sim_obj.interaction_condition) & sim_obj.mask_upper)
    reset_indices_i, reset_indices_j = torch.where(reset_condition)[0], torch.where(reset_condition)[1]

    sim_obj.lifetimes[reset_indices_i, reset_indices_j] += sim_obj.running_lifetimes[reset_indices_i, reset_indices_j]
    sim_obj.running_lifetimes[reset_indices_i, reset_indices_j] = 0
    sim_obj.completed_lifetimes[reset_indices_i, reset_indices_j] += 1

    # Finalize statistics
    if sim_obj.t == sim_obj.t_total - 1:
        for k in range(len(sim_obj.triu_indices[0])):
            i, j = sim_obj.triu_indices[0][k], sim_obj.triu_indices[1][k]
            idx = j - i
            sim_obj.average_lifetimes[idx] += sim_obj.lifetimes[i, j] / (sim_obj.completed_lifetimes[i, j] + 1e-7)
    return None

def update_states(sim_obj):
    t = sim_obj.t - sim_obj.t_half
    if t%10 == 0:
        t_tenth = int(t / 10)
        sim_obj.state_statistics[0, t_tenth] = (sim_obj.states == 0).sum()
        sim_obj.state_statistics[1, t_tenth] = (sim_obj.states == 1).sum()
        sim_obj.state_statistics[2, t_tenth] = (sim_obj.states == 2).sum()

    return None

def update_distances_to_com(sim_obj):
    t = sim_obj.t - sim_obj.t_half

    if t%10 == 0:
        t_tenth = int(t / 10)
        sim_obj.summed_distance_vecs_to_com += (sim_obj.X - sim_obj.center_of_mass)
        norms = torch.linalg.norm(sim_obj.summed_distance_vecs_to_com, dim=1)
        sim_obj.distances_to_com[t_tenth] = float(torch.sum(norms))

    return None

def _gather_statistics(sim_obj):
    # Update radius of gyration
    #update_rg(sim_obj)

    # # Interaction only applies to distances lower than l_interacting
    # # Relevant interactions are only counted once
    # sim_obj.interaction_condition = (sim_obj.interaction_mask == True) & (sim_obj.norms_all < sim_obj.l_interacting) & sim_obj.mask_upper
    # sim_obj.interaction_indices_i = torch.where(sim_obj.interaction_condition)[0]
    # sim_obj.interaction_indices_j = torch.where(sim_obj.interaction_condition)[1]
    #
    # # Count interaction distances
    # update_interaction_idx_differences(sim_obj)
    #
    # # Count correlations by shift
    # update_correlation_sums(sim_obj)
    #
    # # Count lifetimes
    # update_average_lifetimes(sim_obj)

    # Count number of particles in each state
    #update_states(sim_obj)

    update_distances_to_com(sim_obj)

    return None