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
            self.sim_obj.grad_on()
            self.sim_obj.update()
            self.sim_obj.count_interactions()

            # Increment time-step
            self.sim_obj.t += 1

            # Only generates a .gif image for every 10th update
            if idx % 100 == 0:
                yield idx

        # Generates 10 identical images at the end of the animation
        for idx in range(10):
            yield None

    def plot(self, X, Y, Z):
        # Print t
        text_str = r'$t = $' + f' {self.sim_obj.t}'
        r = self.sim_obj.r_system
        self.ax_anim.text(2*r, -1*r, 4*r, text_str)

        # Plot chain
        self.ax_anim.plot(X, Y, Z, lw=0.7, ls='solid', c='k')

        # Plot each state type separately
        for i in range(len(self.sim_obj.states)):
            x_plot = X[self.sim_obj.states[i]]
            y_plot = Y[self.sim_obj.states[i]]
            z_plot = Z[self.sim_obj.states[i]]
            self.ax_anim.scatter(x_plot, y_plot, z_plot, s=self.sim_obj.nucleosome_s, c=self.sim_obj.state_colors[i],
                                 label=self.sim_obj.state_names[i])

        # # Plot polarity vectors
        # u, v, w = self.sim_obj.P[:,0], self.sim_obj.P[:,1], self.sim_obj.P[:,2]
        # self.ax_anim.quiver(X, Y, Z, u, v, w, length=1, normalize=True)

        # Set plot dimensions
        self.ax_anim.set(xlim=self.sim_obj.plot_dim, ylim=self.sim_obj.plot_dim, zlim=self.sim_obj.plot_dim)

        # Set title, labels and legend
        self.ax_anim.set_title(f'No. of nucleosomes = {self.sim_obj.N}', size=16)
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
