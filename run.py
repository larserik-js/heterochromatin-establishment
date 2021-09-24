import pickle

import matplotlib.pyplot as plt
import torch
import copy

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

# Runs the script
def run(N, spring_strength, l0, noise, potential_weights, dt, t_total, test_mode, animate):

    # Fix seed value
    torch.manual_seed(0)

    # Create simulation object
    sim_obj = Simulation(N, spring_strength, l0, noise, potential_weights, dt, t_total)

    # Save initial state for plotting
    x_init = copy.deepcopy(sim_obj.X[:,0])
    y_init = copy.deepcopy(sim_obj.X[:,1])
    z_init = copy.deepcopy(sim_obj.X[:,2])

    coords_init = [x_init, y_init, z_init]

    # Simulation loop
    print('Simulation started.')

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
            filename = '/home/lars/Documents/masters_thesis/animations/animation' + f'_N={sim_obj.N}' + '.gif'
            anim.save(filename, dpi=200, writer=writergif)

            ## Save final state and statistics
            # Final state
            x_final, y_final, z_final = sim_obj.X[:, 0], sim_obj.X[:, 1], sim_obj.X[:, 2]
            # u_final, v_final, w_final = sim_obj.P[:, 0], sim_obj.P[:, 1], sim_obj.P[:, 2]
            # pickle_var_list = [x_final, y_final, z_final, u_final, v_final, w_final, sim_obj.states]
            pickle_var_list = [x_final, y_final, z_final, sim_obj.states,
                               sim_obj.state_colors, sim_obj.state_names, sim_obj.plot_dim]

            filename = 'final_state/final_state' + f'_N={sim_obj.N}'
            write_pkl(pickle_var_list, filename)

            # Statistics
            filename = 'statistics/statistics' + f'_N={sim_obj.N}'
            pickle_var_list = [sim_obj.N, sim_obj.n_interacting, sim_obj.interaction_stats,
                               sim_obj.interaction_idx_difference]
            write_pkl(pickle_var_list, filename)

    # If test_mode, just plot without saving:
    elif test_mode:
        # Iterate
        for t in range(t_total):
            # Print progress
            if (t + 1) % (t_total / 10) == 0:
                print(f'Time-step: {t + 1} / {t_total}')

            # Update
            sim_obj.update()

            # Increment no. of time-steps
            sim_obj.t += 1

        ## Make figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot initial state
        with torch.no_grad():
            sim_obj.plot(x_init, y_init, z_init, ax, label='Initial state', ls='--')

        # Plot final state
        x_final, y_final, z_final = sim_obj.X[:,0], sim_obj.X[:,1], sim_obj.X[:,2]
        #u_final, v_final, w_final = sim_obj.P[:, 0], sim_obj.P[:, 1], sim_obj.P[:, 2]

        with torch.no_grad():
            sim_obj.plot(x_final, y_final, z_final, ax, label='Final state')

        # Plot MD
        sim_obj.finalize_plot(ax)

        # Plot statistics
        sim_obj.plot_statistics()

##############################################################################
##############################################################################