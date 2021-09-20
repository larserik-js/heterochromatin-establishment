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
def run(N, spring_strength, l0, noise, potential_weights, dt, t_total, animate, save):

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
                             interval=100, save_count=t_total+10)
        # Format
        writergif = animation.PillowWriter(fps=30)
        # filename = '/home/lars/Documents/masters_thesis/animation' + f'_N{sim_obj.N}' + '.gif'
        filename = '/home/lars/Documents/masters_thesis/test.gif'

        anim.save(filename, dpi=200, writer=writergif)

    else:
        ## Make figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot initial state
        with torch.no_grad():
            sim_obj.plot(x_init, y_init, z_init, ax, label='Initial state', ls='--')

        for t in range(t_total):
            # Print progress
            if (t+1)%(t_total/10) == 0:
                print(f'Time-step: {t+1} / {t_total}')

            # Update
            sim_obj.grad_on()
            sim_obj.update()
            sim_obj.count_interactions()

            # Increment no. of time-steps
            sim_obj.t += 1

        # Plot final state
        x_final, y_final, z_final = sim_obj.X[:,0], sim_obj.X[:,1], sim_obj.X[:,2]
        u_final, v_final, w_final = sim_obj.P[:, 0], sim_obj.P[:, 1], sim_obj.P[:, 2]

        with torch.no_grad():
            sim_obj.plot(x_final, y_final, z_final, ax, label='Final state')

        # Plot MD
        sim_obj.finalize_plot(ax)

        ## Save final state
        if save:
            pickle_var_list = [x_final, y_final, z_final, u_final, v_final, w_final, sim_obj.states]
            filename = 'final_state' + f'_N{sim_obj.N}'
            write_pkl(pickle_var_list, filename)

            # Save MD plot
            filename = '/home/lars/Documents/masters_thesis/initial_final' + f'_N{sim_obj.N}'
            fig.savefig(filename, dpi=200)

        # Plot statistics
        sim_obj.plot_statistics()

##############################################################################
##############################################################################