# This script is used to import all initial polymer positions for different pressures
# and calculate the RMS, and then write that value to a .txt document.
# The .txt document is located in the 'masters_thesis/pressure_rms/' folder, and is called 'pressure_RMS.txt'.
# These values can be plotted against each other (bottom of script).

import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

from formatting import get_directories


project_dir, _, _ = get_directories()
input_dir = project_dir + 'input/quasi_random_initial_states_pressure_before_dynamics/'
output_file = project_dir + 'pressure_rms/pressure_RMS.txt'
pressure_vals = np.arange(0,1.01,0.01)


def get_rms(X):
    N = X.shape[0]

    # Center-of-mass
    center_of_mass = torch.sum(X, dim=0) / N

    # All distance vectors from the monomers to the center of mass
    dist_vecs_to_com = X - center_of_mass

    # Distances from the monomers to the center of mass
    dist_to_com = torch.norm(dist_vecs_to_com, dim=1)

    # RMS
    rms = torch.sqrt(torch.mean(torch.square(dist_to_com)))

    return rms


def write_file():
    for i, pressure in enumerate(pressure_vals):

        rms = 0
        N_FILES = 100
        for seed in np.arange(N_FILES):
            file_name = input_dir + f'pressure={pressure:.2f}/seed={seed}.pkl'

            ## Open file
            with open(file_name, 'rb') as f:
                X = pickle.load(f)

            rms += get_rms(X)

        # Get the ensemble mean
        mean_rms = rms/N_FILES

        # Append to the .txt file
        line_str = f'{pressure},{mean_rms}' + '\n'

        if i == 0:
            data_file = open(output_file, 'w')
        else:
            data_file = open(output_file, 'a')
        data_file.write(line_str)
        data_file.close()


#write_file()
data = np.loadtxt(output_file, delimiter=',')
plt.plot(data[:,0], data[:,1])
plt.xlabel('Pressure', size=12)
plt.ylabel('RMS', size=12)
plt.show()

#
# import glob
# for fname in glob.glob('/home/lars/PycharmProjects/masters_thesis/quasi_random_initial_states_pressure_before_dynamics/pressure=0.19/*.pkl'):
#     with open(fname, 'rb') as f:
#         x = pickle.load(f)
#     x0 = torch.mean(x, dim=0)
#
#     p_interaction = torch.mean(1.0 * (torch.sqrt(torch.sum((x[None, :, :] - x[:, None, :])**2, dim=2)) < 2))
#     print(p_interaction)
#
# plt.plot(x[:, 0], x[:, 1], '-o')
#
# plt.show()
