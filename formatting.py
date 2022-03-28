import os

# Path to project folder
def get_project_dir():
    project_dir = os.path.abspath(os.path.dirname(__file__)) + '/'

    return project_dir

# Path to output directory
# The directories are located on different locations on local and Cell computers
def get_output_dir(project_dir, run_on_cell=False):
    if run_on_cell:
        output_dir = project_dir + '../../../nbicmplx/cell/zfj803/output/'
    else:
        output_dir = project_dir + 'output/'

    return output_dir

# For filenames
def create_param_string(model, rms, initial_state, cenH_size, cenH_init_idx, ATF1_idx, cell_division, N,
                        t_total, noise, alpha_1, alpha_2, beta, seed, exclude_seed=False):

    param_string = f'{model}_rms={rms:.3f}_init_state={initial_state}_'

    if cell_division:
        param_string += 'cell_division_'

    param_string += f'cenH={cenH_size}_'

    if cenH_size != 0:
        param_string += f'cenH_init_idx={cenH_init_idx}_'

    if ATF1_idx is not None:
        param_string += f'ATF1_idx={ATF1_idx}_'

    param_string += f'N={N}_t_total={t_total}_noise={noise:.4f}_alpha_1={alpha_1:.5f}_alpha_2={alpha_2:.5f}_'\
                    + f'beta={beta:.5f}'

    if exclude_seed == False:
        param_string += f'_seed = {seed}'

    return param_string

def edit_stable_silent_times_file(output_dir, model, rms, initial_state, cenH_size, cenH_init_idx, ATF1_idx,
                                  cell_division, N, t_total, noise, alpha_1, alpha_2, beta, seed, line_str,
                                  action='a'):

    write_name = output_dir + 'statistics/stable_silent_times/'
    write_name += create_param_string(model, rms, initial_state, cenH_size, cenH_init_idx, ATF1_idx,
                                      cell_division, N, t_total, noise, alpha_1, alpha_2, beta,
                                      seed, exclude_seed=True) + '.txt'

    # Append to the file
    data_file = open(write_name, action)
    data_file.write(line_str + '\n')
    data_file.close()

def create_plot_title(model, rms, cenH_size, cenH_init_idx, ATF1_idx, N, t_total, noise, alpha_1, alpha_2, beta, seed):

    param_string = f'"{model}", RMS = {rms:.3f}, cenH = {cenH_size}, '

    if cenH_size != 0:
        param_string += f'cenH_indices = {cenH_init_idx}...{cenH_init_idx + cenH_size - 1}, '

    if ATF1_idx is not None:
        param_string += f'ATF1_idx = {ATF1_idx}, '

    param_string += r'$N$' + f' = {N}, ' + r'$t_{total}$' + f' = {t_total}, noise = {noise:.2f}'\
                    + r'$l_0$' + ', ' + r'$\alpha_1$' + f' = {alpha_1:.5f}, ' + r'$\alpha_2$' + f' = {alpha_2:.5f}, '\
                    + r'$\beta$' + f' = {beta:.5f}, seed = {seed}'

    return param_string

# Make one directory, if it does not already exist
def make_directory(folder_name):
    try:
        os.mkdir(folder_name)
    except FileExistsError:
        pass

# Make all output directories before starting simulations
def make_output_directories(output_dir):
    statistics_folder_names = ['correlation', 'correlation_times', 'dist_vecs_to_com', 'final_state', 'interactions',
                               'Rs', 'states', 'states_time_space', 'successful_conversions', 'stable_silent_times',
                               'optimization']

    make_directory(output_dir)
    make_directory(output_dir + 'animations')
    make_directory(output_dir + 'statistics')

    for folder_name in statistics_folder_names:
        make_directory(output_dir + 'statistics/' + folder_name)