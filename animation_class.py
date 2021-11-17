import matplotlib.pyplot as plt
import torch

class Animation:
    def __init__(self, sim_obj, t_total, coords_init):
        # Parameters
        self.sim_obj = sim_obj
        self.t_total = t_total
        self.coords_init = coords_init

        # Make figure
        self.fig_anim = plt.figure()
        self.ax_anim = self.fig_anim.add_subplot(111, projection='3d')

    def frame_generator(self):
        # Updates the simulation
        for idx in range(self.t_total):

            # Print progress
            if (idx+1)%(self.t_total/10) == 0:
                print(f'Time-step: {idx+1} / {self.t_total}')

            # Update
            self.sim_obj.update()

            # Increment time-step
            self.sim_obj.t = idx + 1

            # Only generates a .gif image for every 100th update
            if idx % 100 == 0:
                yield idx

        # Generates 10 identical images at the end of the animation
        for idx in range(10):
            yield None

    def plot(self, X, Y, Z):
        # Print t
        text_str = r'$t = $' + f' {self.sim_obj.t} / {self.sim_obj.t_total}'
        r = self.sim_obj.r_system
        com = self.sim_obj.center_of_mass
        self.ax_anim.text(r + com[0], -r + com[1], 1.8*r + com[2], text_str)

        # Plot chain
        self.ax_anim.plot(X, Y, Z, lw=0.7, ls='solid', c='k')

        # Plot each state type separately
        for i in range(len(self.sim_obj.states_booleans)):
            x_plot = X[self.sim_obj.states_booleans[i]]
            y_plot = Y[self.sim_obj.states_booleans[i]]
            z_plot = Z[self.sim_obj.states_booleans[i]]
            self.ax_anim.scatter(x_plot, y_plot, z_plot, s=self.sim_obj.nucleosome_s, c=self.sim_obj.state_colors[i],
                                 label=self.sim_obj.state_names[i])

        # # Plot center of mass
        # self.ax_anim.scatter(self.sim_obj.center_of_mass[0], self.sim_obj.center_of_mass[1], self.sim_obj.center_of_mass[2],
        #                      s=0.5, c='g')

        # Set plot dimensions
        self.ax_anim.set(xlim=(self.sim_obj.center_of_mass[0] + self.sim_obj.plot_dim[0],
                               self.sim_obj.center_of_mass[0] + self.sim_obj.plot_dim[1]),
                         ylim=(self.sim_obj.center_of_mass[1] + self.sim_obj.plot_dim[0],
                               self.sim_obj.center_of_mass[1] + self.sim_obj.plot_dim[1]),
                         zlim=(self.sim_obj.center_of_mass[2] + self.sim_obj.plot_dim[0],
                               self.sim_obj.center_of_mass[2] + self.sim_obj.plot_dim[1]))

        # Set title, labels and legend
        self.ax_anim.set_title(r'$N$' + f' = {self.sim_obj.N}, noise = {self.sim_obj.noise}' + r'$l_0$', size=16)
        self.ax_anim.set_title(r'$N$' + f' = {self.sim_obj.N}, ' + r'$t_{total}$' + f' = {self.sim_obj.t_half:.0f},'
                               + f'noise = {self.sim_obj.noise}' + r'$l_0$' + ', ' + r'$\alpha_1$'
                               + f' = {self.sim_obj.alpha_1:.2f}, ' + r'$\alpha_2$' + f' = {self.sim_obj.alpha_2:.2f}', size=16)
        self.ax_anim.set_xlabel('x', size=14)
        self.ax_anim.set_ylabel('y', size=14)
        self.ax_anim.set_zlabel('z', size=14)
        self.ax_anim.legend(loc='upper left')

    def initial_frame(self):
        # Unpack initial state coordinates
        x_init, y_init, z_init = self.coords_init

        self.plot(x_init, y_init, z_init)

    def animation_loop(self, t):
        # Print progress
        if t != None:
            if (t + 1) % (self.t_total / 10) == 0:
                print(f'Time-step: {t + 1} / {self.t_total}')

        with torch.no_grad():
            # Plot
            X_plot, Y_plot, Z_plot = self.sim_obj.X[:, 0].numpy(), self.sim_obj.X[:, 1].numpy(), self.sim_obj.X[:, 2].numpy()

            self.ax_anim.clear()

            self.plot(X_plot, Y_plot, Z_plot)
