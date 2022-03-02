# This script is used to import all initial polymers positions for different pressures
# and calculate the RMS, and then write that value to a .txt document.
# The .txt document is located in the 'masters_thesis' folder, and is called 'pressure_RMS.txt'.
# These values can be plotted against each other (bottom of script).

from formatting import get_project_folder
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt

pathname = get_project_folder()
folder_name = pathname + 'quasi_random_initial_states_pressure_before_dynamics/'
write_name = pathname + 'pressure_RMS.txt'
pressure_vals = np.arange(0,1.01,0.01)


def get_RMS(X):
    N = X.shape[0]

    # Center-of-mass
    COM = torch.sum(X, dim=0) / N

    # All distance vectors from the nucleosomes to the center of mass
    dist_vecs_to_com = X - COM

    # Distances from the nucleosomes to the center of mass
    dist_to_com = torch.norm(dist_vecs_to_com, dim=1)

    # RMS
    RMS = torch.sqrt(torch.mean(torch.square(dist_to_com)))

    return RMS

def write_file():
    for i, pressure in enumerate(pressure_vals):

        RMS = 0
        n_files = 100
        for seed in np.arange(n_files):
            file_name = folder_name + f'pressure={pressure:.2f}/seed={seed}.pkl'

            ## Open file
            with open(file_name, 'rb') as f:
                X = pickle.load(f)

            RMS += get_RMS(X)

        # Get the ensemble mean
        mean_RMS = RMS/n_files

        # Append to the .txt file
        line_str = f'{pressure},{mean_RMS}' + '\n'

        if i == 0:
            data_file = open(write_name, 'w')
        else:
            data_file = open(write_name, 'a')
        data_file.write(line_str)
        data_file.close()

#write_file()

data = np.loadtxt(pathname + '/pressure_RMS.txt', delimiter=',')
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
