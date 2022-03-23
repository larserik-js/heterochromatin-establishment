# External libraries
from timeit import default_timer as timer
import torch
from torch.multiprocessing import Pool
import numpy as np
from datetime import datetime
from functools import partial

# Own modules
import run
from pressure_rms import get_pressure
from formatting import get_project_dir, get_output_dir, make_output_directories, edit_stable_silent_times_file

# Import all parameters
from parameters import n_processes, pool_size, multiprocessing_parameter, animate, set_seed, min_seed, N,l0, noise, dt,\
                       t_total, stats_t_interval, rms, U_two_interaction_weight, allow_state_change, initial_state,\
                       cell_division, cenH_size, cenH_init_idx, write_cenH_data, ATF1_idx, constant, alpha_1,\
                       alpha_1_const, alpha_2, beta

# Get U_pressure_weight value from rms
U_pressure_weight = get_pressure.get_pressure(rms)

##############################################################################
##############################################################################

def curied_run(x, project_dir, output_dir, multiprocessing_parameter, N, l0, noise, dt, t_total,
               U_two_interaction_weight, U_pressure_weight, alpha_1, alpha_2, beta, stats_t_interval, set_seed,
               min_seed, animate, allow_state_change, initial_state, cell_division, cenH_size, cenH_init_idx,
               write_cenH_data, ATF1_idx):


    if multiprocessing_parameter == 'seed':
        return run.run(project_dir, output_dir, N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight,
                       alpha_1, alpha_2, beta, stats_t_interval, set_seed, x, animate, allow_state_change,
                       initial_state, cell_division, cenH_size, cenH_init_idx, write_cenH_data, ATF1_idx)

    elif multiprocessing_parameter == 'alpha_1':
        return run.run(project_dir, output_dir, N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight,
                       x, alpha_2, beta, stats_t_interval, set_seed, min_seed, animate, allow_state_change,
                       initial_state, cell_division, cenH_size, cenH_init_idx, write_cenH_data, ATF1_idx)

    elif multiprocessing_parameter == 'rms':
        return run.run(project_dir, output_dir, N, l0, noise, dt, t_total, U_two_interaction_weight, x, alpha_1,
                       alpha_2, beta, stats_t_interval, set_seed, min_seed, animate, allow_state_change, initial_state,
                       cell_division, cenH_size, cenH_init_idx, write_cenH_data, ATF1_idx)
    else:
        raise AssertionError('Invalid multiprocessing_parameter given.')

def main(run_on_cell=False, n_processes=n_processes, pool_size=pool_size, N=N, l0=l0, noise=noise, dt=dt,
         t_total=t_total, U_two_interaction_weight=U_two_interaction_weight, U_pressure_weight=U_pressure_weight,
         alpha_1=alpha_1, alpha_2=alpha_2, beta=beta, stats_t_interval=stats_t_interval, set_seed=set_seed,
         min_seed=min_seed, animate=animate, allow_state_change=allow_state_change, initial_state=initial_state,
         cell_division=cell_division, cenH_size=cenH_size, cenH_init_idx=cenH_init_idx,
         write_cenH_data=write_cenH_data, ATF1_idx=ATF1_idx):

    # Get paths to project and output directories
    project_dir = get_project_dir()
    output_dir = get_output_dir(project_dir, run_on_cell)

    # Make output directories
    make_output_directories(output_dir)

    # Get detailed error messages
    torch.autograd.set_detect_anomaly(False)

    # Start the timer
    print(f'Simulation started.')
    initial_time = timer()

    # Parameter arrays for parallel processes
    # Seed list
    if multiprocessing_parameter == 'seed':
        parameter_list = np.arange(n_processes) + min_seed
    # alpha_1 list
    elif multiprocessing_parameter == 'alpha_1':
        parameter_list = np.linspace(25, 50, n_processes) * 0.02 * 0.1 * alpha_1_const
    # U_pressure_weight list
    elif multiprocessing_parameter == 'rms':
        parameter_list = np.linspace(0,1,n_processes)
    # constant list
    elif multiprocessing_parameter == 'constant':
        parameter_list = constant * np.logspace(start=-2, stop=0, num=n_processes)
    else:
        raise AssertionError('Invalid multiprocessing parameter given.')

    # Multiprocessing
    if n_processes > 1:
        torch.set_num_threads(1)

        # Make the file for cenH statistics
        if (cenH_size > 0) and write_cenH_data:
            # Write the file, and the first two lines
            line_str = f't_total={t_total}' + '\n' + 'silent_t,half_silent_t,n_patches,seed'
            edit_stable_silent_times_file(output_dir, U_pressure_weight, initial_state, cenH_size, cenH_init_idx,
                                          ATF1_idx, cell_division, N, t_total, noise, alpha_1, alpha_2, beta, min_seed,
                                          line_str, action='w')
        else:
            pass

        # Create pool for multiprocessing
        pool = Pool(pool_size)

        res = list(pool.map(partial(curied_run, project_dir=project_dir, output_dir=output_dir,
                                    multiprocessing_parameter=multiprocessing_parameter, N=N, l0=l0, noise=noise, dt=dt,
                                    t_total=t_total, U_two_interaction_weight=U_two_interaction_weight,
                                    U_pressure_weight=U_pressure_weight, alpha_1=alpha_1, alpha_2=alpha_2, beta=beta,
                                    stats_t_interval=stats_t_interval, set_seed=set_seed, min_seed=min_seed,
                                    animate=animate, allow_state_change=allow_state_change, initial_state=initial_state,
                                    cell_division=cell_division, cenH_size=cenH_size, cenH_init_idx=cenH_init_idx,
                                    write_cenH_data=write_cenH_data, ATF1_idx=ATF1_idx
                                    ),
                            parameter_list, chunksize=1
                            )
                   )

    # Run a single process without multiprocessing
    elif n_processes == 1:
        res = list(map(partial(curied_run, project_dir=project_dir, output_dir=output_dir,
                               multiprocessing_parameter=multiprocessing_parameter, N=N, l0=l0, noise=noise, dt=dt,
                               t_total=t_total, U_two_interaction_weight=U_two_interaction_weight,
                               U_pressure_weight=U_pressure_weight, alpha_1=alpha_1, alpha_2=alpha_2, beta=beta,
                               stats_t_interval=stats_t_interval, set_seed=set_seed, min_seed=min_seed, animate=animate,
                               allow_state_change=allow_state_change, initial_state=initial_state,
                               cell_division=cell_division, cenH_size=cenH_size, cenH_init_idx=cenH_init_idx,
                               write_cenH_data=write_cenH_data, ATF1_idx=ATF1_idx
                               ),
                       parameter_list
                       )
                   )

        # Print time elapsed
        final_time = timer()-initial_time
        print(f'Simulation finished at {final_time:.2f} s, {datetime.now()}')

    else:
        raise AssertionError("n_processes set to 0. Choose a higher value.")

    return res


## RUN THE SCRIPT
if __name__ == '__main__':
    _ = main(run_on_cell=False, n_processes=n_processes, pool_size=pool_size, N=N, l0=l0, noise=noise, dt=dt,
             t_total=t_total, U_two_interaction_weight=U_two_interaction_weight, U_pressure_weight=U_pressure_weight,
             alpha_1=alpha_1, alpha_2=alpha_2, beta=beta, stats_t_interval=stats_t_interval, set_seed=set_seed,
             min_seed=min_seed, animate=animate, allow_state_change=allow_state_change, initial_state=initial_state,
             cell_division=cell_division, cenH_size=cenH_size, cenH_init_idx=cenH_init_idx,
             write_cenH_data=write_cenH_data, ATF1_idx=ATF1_idx)


