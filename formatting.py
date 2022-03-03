import os
import sys

# # Path to the project folder
# pathname = os.path.abspath(os.path.dirname(__file__)) + '/'

statistics_folder_names = ['correlation', 'correlation_times', 'dist_vecs_to_com', 'final_state', 'interactions', 'Rs',
                           'states', 'states_time_space', 'successful_conversions', 'stable_silent_times',
                           'optimization']

# Path to folder to save files
def get_project_folder(run_on_cell=False):
    pathname = os.path.abspath(os.path.dirname(__file__)) + '/'

    if run_on_cell:
        pathname += '../../../nbicmplx/cell/zfj803/'
    else:
        pass
    return pathname

def create_param_string(U_pressure_weight, initial_state, cenH_size, cenH_init_idx, cell_division, barriers, N, t_total, noise, alpha_1,
                          alpha_2, beta, seed, exclude_seed=False):

    param_string = f'pressure={U_pressure_weight:.2f}_init_state={initial_state}_'

    if cell_division:
        param_string += 'cell_division_'
    if barriers:
        param_string += 'barriers_'

    param_string += f'cenH={cenH_size}_cenH_init_idx={cenH_init_idx}_N={N}_t_total={t_total}_noise={noise:.4f}'\
                    + f'_alpha_1={alpha_1:.5f}_alpha_2={alpha_2:.5f}_beta={beta:.5f}'

    if exclude_seed == False:
        param_string += f'_seed = {seed}'
    else:
        pass

    return param_string

def edit_stable_silent_times_file(pathname, U_pressure_weight, initial_state, cenH_size, cenH_init_idx, cell_division,
                                  barriers, N, t_total, noise, alpha_1, alpha_2, beta, seed, line_str, action='a'):

    write_name = pathname + 'data/statistics/stable_silent_times/stable_silent_times_'
    write_name += create_param_string(U_pressure_weight, initial_state, cenH_size, cenH_init_idx,
                                      cell_division, barriers, N, t_total, noise, alpha_1, alpha_2, beta,
                                      seed, exclude_seed=True) + '.txt'

    # Append to the file
    data_file = open(write_name, action)
    data_file.write(line_str + '\n')
    data_file.close()

def create_plot_title(U_pressure_weight, cenH_size, cenH_init_idx, barriers, N, t_total, noise, alpha_1, alpha_2, beta, seed):
    param_string = ''

    if barriers:
        param_string += 'barriers, '

    param_string += f'pressure = {U_pressure_weight:.2f}, cenH = {cenH_size}, '\
                    + f'cenH_indices = {cenH_init_idx}...{cenH_init_idx + cenH_size - 1}, '\
                    + r'$N$' + f' = {N}, ' + r'$t_{total}$' + f' = {t_total}, noise = {noise:.2f}'\
                    + r'$l_0$' + ', ' + r'$\alpha_1$' + f' = {alpha_1:.5f}, ' + r'$\alpha_2$' + f' = {alpha_2:.5f}, '\
                    + r'$\beta$' + f' = {beta:.5f}, seed = {seed}'

    return param_string

def create_directories(pathname):
    try:
        os.mkdir(pathname + 'data')
    except FileExistsError:
        pass

    try:
        os.mkdir(pathname + 'data/statistics')
    except FileExistsError:
        pass

    for folder_name in statistics_folder_names:
        try:
            os.mkdir(pathname + 'data/statistics/' + folder_name)
        except FileExistsError:
            pass
