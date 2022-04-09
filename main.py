from timeit import default_timer as timer
from datetime import datetime
from functools import partial

import torch
from torch.multiprocessing import Pool
import numpy as np

import run
from formatting import (get_directories, make_output_directories,
                        edit_stable_silent_times_file)
#from parameters import *
from parameters import params


def curied_run(x, model, project_dir, input_dir, output_dir,
               multiprocessing_param, N, l0, noise, dt, t_total,
               interaction_size, rms, alpha_1, alpha_2, beta, set_seed,
               min_seed, animate, allow_state_change, initial_state,
               cell_division, cenH_size, cenH_init_idx, write_cenH_data,
               ATF1_idx):

    if multiprocessing_param == 'seed':
        return run.run(model, project_dir, input_dir, output_dir, N, l0, noise,
                       dt, t_total, interaction_size, rms, alpha_1, alpha_2,
                       beta, set_seed, x, animate, allow_state_change,
                       initial_state, cell_division, cenH_size, cenH_init_idx,
                       write_cenH_data, ATF1_idx)

    elif multiprocessing_param == 'alpha_1':
        return run.run(model, project_dir, input_dir, output_dir, N, l0, noise,
                       dt, t_total, interaction_size, rms, x, alpha_2, beta,
                       set_seed, min_seed, animate, allow_state_change,
                       initial_state, cell_division, cenH_size, cenH_init_idx,
                       write_cenH_data, ATF1_idx)

    elif multiprocessing_param == 'rms':
        return run.run(model, project_dir, input_dir, output_dir, N, l0, noise,
                       dt, t_total, interaction_size, x, alpha_1, alpha_2, beta,
                       set_seed, min_seed, animate, allow_state_change,
                       initial_state, cell_division, cenH_size, cenH_init_idx,
                       write_cenH_data, ATF1_idx)
    else:
        raise AssertionError('Invalid multiprocessing_param given.')


def main(*, model, run_on_cell, n_processes, pool_size, multiprocessing_param, 
         N, l0, noise, dt, t_total, interaction_size, rms, alpha_1, alpha_2,
         beta, alpha_1_const, set_seed, min_seed, animate, allow_state_change,
         initial_state, cell_division, cenH_size, cenH_init_idx,
         write_cenH_data, ATF1_idx):

    # Get paths to project, input, and output directories
    project_dir, input_dir, output_dir = get_directories(run_on_cell)

    # Make output directories
    make_output_directories(output_dir)

    # Get detailed error messages
    torch.autograd.set_detect_anomaly(False)

    # Start the timer
    print(f'Simulation started.')
    initial_time = timer()

    # Parameter arrays for parallel processes
    # Seed list
    if multiprocessing_param == 'seed':
        parameter_list = np.arange(n_processes) + min_seed
    # alpha_1 list
    elif multiprocessing_param == 'alpha_1':
        parameter_list = (np.linspace(25, 50, n_processes) * 0.02 * 0.1
                          * alpha_1_const)
    # rms list
    elif multiprocessing_param == 'rms':
        parameter_list = np.linspace(1.677, 4.130, n_processes)
    else:
        raise AssertionError('Invalid multiprocessing parameter given.')

    # Multiprocessing
    if n_processes > 1:
        torch.set_num_threads(1)

        # Make the file for cenH statistics
        if (cenH_size > 0) and write_cenH_data:
            # Write the file, and the first two lines
            line_str = (f't_total={t_total}' + '\n'
                        + 'silent_t,half_silent_t,n_patches,seed')
            edit_stable_silent_times_file(
                output_dir, model, rms, initial_state, cenH_size, cenH_init_idx,
                ATF1_idx, cell_division, N, t_total, noise, alpha_1, alpha_2,
                beta, min_seed, line_str, action='w')
        else:
            pass

        # Create pool for multiprocessing
        pool = Pool(pool_size)

        res = list(pool.map(
            partial(curied_run, model=model, project_dir=project_dir,
                    input_dir=input_dir, output_dir=output_dir,
                    multiprocessing_param=multiprocessing_param, N=N,
                    l0=l0, noise=noise, dt=dt, t_total=t_total,
                    interaction_size=interaction_size, rms=rms, alpha_1=alpha_1,
                    alpha_2=alpha_2, beta=beta, set_seed=set_seed,
                    min_seed=min_seed, animate=animate,
                    allow_state_change=allow_state_change,
                    initial_state=initial_state, cell_division=cell_division,
                    cenH_size=cenH_size, cenH_init_idx=cenH_init_idx,
                    write_cenH_data=write_cenH_data, ATF1_idx=ATF1_idx),
            parameter_list, chunksize=1)
        )

    # Run a single process without multiprocessing
    elif n_processes == 1:
        res = list(map(
            partial(curied_run, model=model, project_dir=project_dir,
                    input_dir=input_dir, output_dir=output_dir,
                    multiprocessing_param=multiprocessing_param, N=N,
                    l0=l0, noise=noise, dt=dt, t_total=t_total,
                    interaction_size=interaction_size, rms=rms, alpha_1=alpha_1,
                    alpha_2=alpha_2, beta=beta, set_seed=set_seed,
                    min_seed=min_seed, animate=animate,
                    allow_state_change=allow_state_change,
                    initial_state=initial_state, cell_division=cell_division,
                    cenH_size=cenH_size, cenH_init_idx=cenH_init_idx,
                    write_cenH_data=write_cenH_data, ATF1_idx=ATF1_idx),
            parameter_list)
        )

        # Print time elapsed
        final_time = timer()-initial_time
        print(f'Simulation finished at {final_time:.2f} s, {datetime.now()}')

    else:
        raise AssertionError("n_processes set to 0. Choose a higher value.")

    return res


if __name__ == '__main__':
    _ = main(**params)