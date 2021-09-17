import copy
from scipy.special import binom

import numpy as np
import torch
import numba as nb
from numba import njit
from timeit import default_timer as DT

N = 6

r = np.random

n_allowed_interactions = 2
shape = (N,N)

#torch.manual_seed(3)
def generate_test_tensor():
    test_array = torch.randn(shape)
    test_array = torch.triu(test_array,diagonal=1)
    test_array += test_array.t()
    test_array = torch.abs(test_array)
    return test_array


@njit
def simple():
    array = np.ones(shape=(N,N),dtype=nb.boolean)
    print(array)
    #array = np.ones((N,N),dtype=bool)

    indices = np.array([1,3],dtype=nb.int64)

    print(indices)
    print(array[indices])
    return
@njit
def set_difference(array_A, array_B):
    return np.array(set(array_A) - set(array_B))

def get_interaction_mask(torch_tensor):

    # Transform torch tensor to numpy array
    full_array = torch_tensor.detach().numpy()

    @njit
    def mask_calculator():
        #array_shape = full_array.shape

        # Picks out allowed interactions,
        # i.e. nucleosomes cannot interact with themselves,
        # and only with a number of other nucleosomes determined by 'n_allowed_interactions'
        allowed_mask = np.ones_like(full_array, dtype=nb.boolean)
#
        #allowed_mask = np.ones(array_shape, dtype=bool)

        np.fill_diagonal(allowed_mask, False)
        np.fill_diagonal(allowed_mask[:-1,1:], False)
        np.fill_diagonal(allowed_mask[1:,:-1], False)

        # Shows which nucleosomes interact with which
        interaction_mask = np.zeros_like(allowed_mask)

        while True:
            counter = 0

            # Minimum allowed value
            # i.e. lowest value of valid interactions
            min_val = full_array.ravel()[allowed_mask.ravel()].min()

            # Indices of these values
            indices = np.where(full_array==min_val)[0]

            # Set interactions as valid
            idx1, idx2 = indices[0], indices[1]
            interaction_mask[idx1, idx2] = True
            interaction_mask[idx2, idx1] = True
            # print('Interaction mask:')
            # print(interaction_mask)

            # Do not check for these interactions after this step
            allowed_mask[idx1, idx2] = False
            allowed_mask[idx2, idx1] = False

            # Total interactions per nucleosome
            interaction_sums = []
            for i in range(N):
                column_sum = 0
                for j in range(N):
                    column_sum += interaction_mask[i,j]

                interaction_sums.append(column_sum)

            interaction_sums = np.array(interaction_sums, dtype=nb.int64)

            # If the nucleosomes at the end of the chain interacts with two other nucleosomes
            # other than itself and their one nearest neighbor
            if interaction_sums[0] == n_allowed_interactions:
                #allowed_mask[0] = False
                #allowed_mask[:,0] = False

                for i in range(N):
                    allowed_mask[0,i] = False
                    allowed_mask[i,0] = False

            if interaction_sums[-1] == n_allowed_interactions:
                #allowed_mask[-1] = False
                #allowed_mask[:,-1] = False

                for i in range(N):
                    allowed_mask[-1,i] = False
                    allowed_mask[i,-1] = False

            # If a nucleosome interacts with two nucleosomes
            # other than itself and its two nearest neighbors
            if np.any(interaction_sums >= n_allowed_interactions):
                saturated_indices = np.where(interaction_sums >= n_allowed_interactions)[0]
                allowed_mask[saturated_indices] = False
                allowed_mask[:, saturated_indices] = False

                for idx in saturated_indices:
                    for i in range(N):
                        allowed_mask[idx,i] = False
                        allowed_mask[i,idx] = False

            counter +=1

            if np.sum(allowed_mask) == 0:
                print(f'Total interactions: {np.sum(interaction_mask)}')
                break

            if counter == 10000:
                break
        return interaction_mask
    return mask_calculator

# #### WITHOUT NUMBA ####
# def get_interaction_mask(torch_tensor):
#     # Transform torch tensor to numpy array
#     full_array = torch_tensor.detach().numpy()
#
#     def mask_calculator():
#         # array_shape = full_array.shape
#
#         # Picks out allowed interactions,
#         # i.e. nucleosomes cannot interact with themselves,
#         # and only with a number of other nucleosomes determined by 'n_allowed_interactions'
#         allowed_mask = np.ones_like(full_array, dtype=bool)
#         #
#         # allowed_mask = np.ones(array_shape, dtype=bool)
#
#         np.fill_diagonal(allowed_mask, False)
#         np.fill_diagonal(allowed_mask[:-1, 1:], False)
#         np.fill_diagonal(allowed_mask[1:, :-1], False)
#
#         # Shows which nucleosomes interact with which
#         interaction_mask = np.zeros_like(allowed_mask)
#
#         while True:
#             counter = 0
#
#             # Minimum allowed value
#             # i.e. lowest value of valid interactions
#
#             min_val = full_array[allowed_mask].min()
#
#             # Indices of these values
#             indices = np.where(full_array == min_val)
#
#             # Set interactions as valid
#             interaction_mask[indices[0], indices[1]] = True
#             # print('Interaction mask:')
#             # print(interaction_mask)
#
#             # Do not check for these interactions after this step
#             allowed_mask[indices[0], indices[1]] = False
#
#             # Total interactions per nucleosome
#             interaction_sums = np.sum(interaction_mask, axis=0)
#
#             # If the nucleosomes at the end of the chain interacts with two other nucleosomes
#             # other than itself and their one nearest neighbor
#             if interaction_sums[0] == n_allowed_interactions:
#                 allowed_mask[0] = False
#                 allowed_mask[:, 0] = False
#             if interaction_sums[-1] == n_allowed_interactions:
#                 allowed_mask[-1] = False
#                 allowed_mask[:, -1] = False
#
#             # If a nucleosome interacts with two nucleosomes
#             # other than itself and its two nearest neighbors
#             if np.any(interaction_sums >= n_allowed_interactions):
#                 saturated_indices = np.where(interaction_sums >= n_allowed_interactions)[0]
#                 allowed_mask[saturated_indices] = False
#                 allowed_mask[:, saturated_indices] = False
#
#             counter += 1
#             if np.sum(allowed_mask) == 0:
#                 print(f'Total interactions: {np.sum(interaction_mask)}')
#                 break
#
#             if counter == 10000:
#                 break
#         return interaction_mask
#
#     return mask_calculator

print('Numba:')
for i in range(10):
    test_array = generate_test_tensor()
    t_init = DT()

    mask_calculator = get_interaction_mask(test_array)

    interaction_mask = mask_calculator()

    print(f'Simulation time: {DT()-t_init:.5f}')

# print('NumPy:')
# for i in range(10):
#     test_array = generate_test_tensor()
#     t_init = DT()
#
#     mask_calculator = get_interaction_mask(test_array)
#
#     interaction_mask = mask_calculator()
#
#     print(f'Simulation time: {DT()-t_init:.5f}')
