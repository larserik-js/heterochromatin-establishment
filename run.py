import pickle
import matplotlib.pyplot as plt
import torch
torch.set_num_threads(1)
import copy
import os

# Animation packages
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# Simulation class
from simulation_class import Simulation

# External animation file
from animation_class import Animation


# Takes a list of torch tensors, pickles them
def write_pkl(var_list, filename):
    filename = '/home/lars/Documents/masters_thesis/' + filename + '.pkl'

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

def save_data(sim_obj):
    ## Save final state and statistics
    # Filenames
    if sim_obj.classic:
        fs_filename =  f'final_state/classic_final_state_N={sim_obj.N}_t_total={sim_obj.t_total}_noise={sim_obj.noise:.2f}'
        stats_filename = f'statistics/classic_statistics_N={sim_obj.N}_t_total={sim_obj.t_total}_noise={sim_obj.noise:.2f}'
        rg_filename = f'statistics/RG/classic_RG_N={sim_obj.N}_t_total={sim_obj.t_total}_noise={sim_obj.noise:.2f}'
    else:
        fs_filename =  f'final_state/non-classic_final_state_N={sim_obj.N}_t_total={sim_obj.t_total}_noise={sim_obj.noise:.2f}'
        stats_filename =  f'statistics/non-classic_statistics_N={sim_obj.N}_t_total={sim_obj.t_total}_noise={sim_obj.noise:.2f}'
        rg_filename = f'statistics/RG/non-classic_RG_N={sim_obj.N}_t_total={sim_obj.t_total}_noise={sim_obj.noise:.2f}'

    # Final state
    x_final, y_final, z_final = sim_obj.X[:, 0], sim_obj.X[:, 1], sim_obj.X[:, 2]
    # u_final, v_final, w_final = sim_obj.P[:, 0], sim_obj.P[:, 1], sim_obj.P[:, 2]
    # pickle_var_list = [x_final, y_final, z_final, u_final, v_final, w_final, sim_obj.states]
    pickle_var_list = [x_final, y_final, z_final, sim_obj.states,
                       sim_obj.state_colors, sim_obj.state_names, sim_obj.plot_dim]
    write_pkl(pickle_var_list, fs_filename)

    # Statistics
    pickle_var_list = [sim_obj.N, sim_obj.noise, sim_obj.n_interacting,
                       sim_obj.interaction_idx_difference, sim_obj.average_lifetimes]
    write_pkl(pickle_var_list, stats_filename)

    # Radius of gyration
    pickle_var_list = [sim_obj.radius_of_gyration]
    write_pkl(pickle_var_list, rg_filename)

# Runs the script
def run(N, spring_strength, l0, noise, potential_weights, dt, t_total, classic, test_mode, animate, verbose):
    torch.set_num_threads(1)
    print(f'Started simulation with noise = {noise}')

    # Fix seed value
    torch.manual_seed(0)

    # Create simulation object
    sim_obj = Simulation(N, spring_strength, l0, noise, potential_weights, dt, t_total, classic)

    # Save initial state for plotting
    x_init = copy.deepcopy(sim_obj.X[:,0])
    y_init = copy.deepcopy(sim_obj.X[:,1])
    z_init = copy.deepcopy(sim_obj.X[:,2])

    coords_init = [x_init, y_init, z_init]

    # Simulation loop
    if animate:
        # Create animation object
        anim_obj = Animation(sim_obj, t_total, coords_init)

        # The animation loop
        # The function 'animation_loop' gets called t_total times
        anim = FuncAnimation(anim_obj.fig_anim, anim_obj.animation_loop, frames=anim_obj.frame_generator,
                             interval=100, save_count=t_total + 10)
        # Format and save
        writergif = animation.PillowWriter(fps=30)

        # Just save a test gif
        if test_mode:
            filename = '/home/lars/Documents/masters_thesis/animations/test.gif'
            anim.save(filename, dpi=200, writer=writergif)

        # Save a named gif, plus pickled final states and statistics
        else:
            ## Save animation
            filename = '/home/lars/Documents/masters_thesis/animations/animation' \
                       + f'_N={sim_obj.N}' + f'_noise={sim_obj.noise:.2f}' + '.gif'
            anim.save(filename, dpi=200, writer=writergif)

            ## Save data
            save_data(sim_obj)

    else:
        # Iterate
        for t in range(t_total):
            # Print progress
            if verbose:
                if (t + 1) % (t_total / 10) == 0:
                    print(f'{os.getpid()} : Time-step: {t + 1} / {t_total}   (noise={noise})')

            # Update
            sim_obj.update()

            # Increment no. of time-steps
            sim_obj.t += 1

        # Just plot without saving:
        if test_mode:
            # ## Make figure
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            #
            # # Plot initial state
            # with torch.no_grad():
            #     sim_obj.plot(x_init, y_init, z_init, ax, label='Initial state', ls='--')
            #
            # # Plot final state
            # x_final, y_final, z_final = sim_obj.X[:,0], sim_obj.X[:,1], sim_obj.X[:,2]
            # #u_final, v_final, w_final = sim_obj.P[:, 0], sim_obj.P[:, 1], sim_obj.P[:, 2]
            #
            # with torch.no_grad():
            #     sim_obj.plot(x_final, y_final, z_final, ax, label='Final state')
            #
            # # Plot MD
            # sim_obj.finalize_plot(ax)
            #
            # # Plot statistics
            # sim_obj.plot_statistics()
            import plotter

        # Just save statistics, no plotting
        else:
            ## Save data
            save_data(sim_obj)

    print(f'Finished simulation with noise = {noise}')

##############################################################################
##############################################################################