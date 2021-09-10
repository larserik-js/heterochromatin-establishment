import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.special import lambertw
import copy

# Animation packages
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# External animation file
from animation_class import Animation

# Classes and functions
class Simulation:
    def __init__(self, parameters):
        ## Parameters
        # Unpack
        N, spring_strength, l0, noise, potential_weights, dt, t_total, _ = parameters

        # No. of nucleosomes
        self.N = N
        # Half the spring constant
        self.spring_strength = spring_strength
        # Equilibrium spring length
        self.l0 = l0
        # Noise term
        self.noise = noise
        # Potential weights
        self.potential_weights = potential_weights

        ## Initialize system
        thetas = np.linspace(0, 2*np.pi, N, endpoint=False)
        r_system = self.N*self.l0 / (2*np.pi)
        xs, ys = r_system*np.cos(thetas), r_system*np.sin(thetas)
        zs = np.linspace(-r_system,r_system,N)

        # Nucleosome positions
        self.X = torch.tensor([xs, ys, zs], dtype=torch.double).t()

        # No. of interacting nucleosomes
        n_interacting = int(N/2)

        # States
        # Set no. of interacting nucleosomes to 0s
        # Set the rest randomly to 1s or 2s
        states = torch.zeros_like(self.X[:,0])
        states[n_interacting:] = torch.randint(1,3,(N-n_interacting,))
        # Pick out the nucleosomes of the different states
        self.state_A = (states==0)
        self.state_B = (states==1)
        self.state_C = (states==2)

        self.states = [self.state_A, self.state_B, self.state_C]

        # Indices of interacting and non-interacting states
        self.interacting_idx = (states == 0).nonzero()
        self.non_interacting_idx = (states != 0).nonzero()

        # Convert to vector, i.e. row dimension = 0
        self.interacting_idx = self.interacting_idx.squeeze()
        self.non_interacting_idx = self.non_interacting_idx.squeeze()

        ## For statistics
        # As a starting point: the interaction distance between nucleosomes is just set to
        # the same distance as the equilibrium spring length
        # The linker DNA in reality consists of up to about 80 bp
        self.r0 = self.l0 * 1

        # Particles within the following distance are counted for statistcs
        self.l_interacting = 2*self.r0

        ## Distance vectors from all particles to all particles
        rij_all = self.X[:, None, :] - self.X

        # Length of distances
        self.norms_all = torch.linalg.norm(rij_all, dim=2)

        # Polarity vectors
        # In this case sampled from a unit normal distribution
        self.P = torch.randn_like(self.X)
        # Normalize polarity vectors
        norms = torch.linalg.norm(self.P, dim=1)
        self.P = self.P / norms[:,None]

        # Used for indexing neighbors
        chain_idx = np.stack([np.roll(np.arange(N), 1),
                              np.roll(np.arange(N), -1)]
                             ).T
        self.chain_idx = torch.tensor(chain_idx, dtype=torch.long)

        # Picks out the neighbors of each nucleosome.
        # The nucleosomes at the polymer ends only have one neighbor.
        self.mask = torch.isfinite(self.chain_idx)
        self.mask[0,0] = 0
        self.mask[-1,1] = 0

        # # For each particle lists all other particles from closest to furthest away
        # self.neighbors = torch.from_numpy(get_neighbors(self.X, k = self.X.shape[0]-1))
        move_to_gpu = False
        if move_to_gpu:
            self.X = self.X.cuda()
            self.mask = self.mask.cuda()

        # No. of time steps
        self.t = 0
        # Time-step
        self.dt = dt
        self.t_total = t_total

        # Will store information on the number of neighbors within a given distance
        # for interacting and non-interacting states, respectively
        self.interaction_stats = torch.zeros((2,t_total))

        ## Plot parameters
        # Nucleosome scatter marker size
        self.nucleosome_s = 5
        # Chain scatter marker size
        self.chain_s = 1
        # Colors of scatter plot markers
        self.state_colors = ['b', 'r', 'y']
        self.state_names = ['State A', 'State B', 'State C']
        # Plot dimensions
        self.plot_dim = (-1.5*r_system, 1.5*r_system)
        self.r_system = r_system

    def grad_on(self):
        self.X.requires_grad_(True)
        #self.P.requires_grad_(True)

    # Returns (overall) system potential
    def potential(self):
        U_spring = self.spring_potential()

        ################################################################################################################

        ## INTERACTION-BASED POTENTIAL
        U_interaction = self.interaction_potential()

        ################################################################################################################

        ## PRESSURE POTENTIAL
        U_pressure = self.pressure_potential()

        w1, w2, w3 = self.potential_weights[0], self.potential_weights[1], self.potential_weights[2]

        if (self.t == 1) | (self.t == self.t_total):
            print(f'Spring potential: {w1*U_spring:.2}')
            print(f'Interaction potential: {w2*U_interaction:.2}')
            print(f'Pressure potential: {w3*U_pressure:.2}')

        return w1*U_spring + w2*U_interaction + w3*U_pressure

    def pressure_potential(self):
        # Enacted by the nuclear envelope
        norms = torch.linalg.norm(self.X, dim=1)
        #U_pressure = torch.sum(torch.exp(norms/2) - 1)
        U_pressure = torch.sum(norms)
        return U_pressure

    # # This version only includes a repulsion term for interacting nucleosomes
    # def interaction_potential(self):
    #     # Based on the fact that the linker DNA consists of up to about 80 bp
    #     r0 = self.l0 / 80
    #     # Only calculate on interacting nucleosomes
    #     X_interacting = self.X[self.state_A]
    #     # Distance vectors from an interacting particle to all other interacting particles in the chain
    #     rij_all = X_interacting[:, None, :] - X_interacting
    #     # Length of distances
    #     norms_all = torch.linalg.norm(rij_all, dim=2)
    #
    #     # Leaves out self-self interactions
    #     mask_diag = (norms_all != 0)
    #     # Normalize distance vectors
    #     rij_all = rij_all / (1e-10 + norms_all[:, :, None])
    #     # Regulate the potential function
    #     # b is the value which ensures that r0 is a local extremum for U_interaction
    #     b = np.real(-2 / lambertw(-2 * np.exp(-2)))
    #     # Sum potential terms
    #     U_repulsion = torch.exp(-2 * norms_all / r0) * mask_diag
    #     U_attraction = -torch.exp(-2 * norms_all / (b * r0)) * mask_diag
    #     U_interaction = torch.sum(U_repulsion + U_attraction)
    #     return U_interaction


    # This version includes a repulsion potential for ALL nucleosomes,
    # not just for state A nucleosomes
    def interaction_potential(self):

        # Normalize distance vectors
        #rij_all = rij_all / (1e-10 + self.norms_all[:, :, None])

        # Regulate the potential function
        # b is the value which ensures that r0 is a local extremum for U_interaction
        b = np.real(-2 / lambertw(-2 * np.exp(-2)))

        U_repulsion = torch.exp(-2 * self.norms_all / self.r0)

        U_attraction = -torch.exp(-2 * self.norms_all / (b * self.r0))

        # Only calculate on interacting nucleosomes
        mask_attracting = self.state_A * self.state_A[:,None]

        U_attraction = U_attraction * mask_attracting

        # Leaves out self-self interactions
        mask_diag = (self.norms_all != 0)

        U_interaction = torch.sum((U_repulsion + U_attraction)[mask_diag])

        return U_interaction

    def spring_potential(self):
        ## SPRING-BASED POTENTIAL
        # Distance vectors from a particle to its two neighbors in the chain
        rij = self.X[self.chain_idx] - self.X[:, None, :]
        # Length of the distance vectors
        norms = torch.linalg.norm(rij, dim=2)
        # print(norms)
        # Normalize distance vectors
        self.rij_hat = rij / (1e-10 + norms[:, :, None])
        # print(self.mask * (norms - self.l0)**2)
        # The spring-based potential term
        U_spring = self.spring_strength * torch.sum(self.mask * (norms - self.l0) ** 2)
        return U_spring

    def update(self):
        # Calculate potential
        U = self.potential()
        U.backward()

        ## Update variables
        with torch.no_grad():
            if torch.isnan(torch.sum(self.X.grad)):
                raise AssertionError('NAN in X gradient!')

            # Positions
            self.X -= self.X.grad * self.dt

            # Add noise
            self.X += self.noise * torch.randn_like(self.X) * np.exp(-self.t / self.t_total)
            #self.X += self.noise * torch.randn_like(self.X) * (1 - self.t/self.t_total)
            #self.X += self.noise * torch.randn_like(self.X)

            # Reset gradients
            self.X.grad.zero_()

        # Distance vectors from all particles to all particles
        rij_all = self.X[:, None, :] - self.X

        # Length of distances
        self.norms_all = torch.linalg.norm(rij_all, dim=2)

        # Increment no. of time-steps
        #self.t += 1

    def count_interactions(self):
        norms_all_upper = torch.triu(self.norms_all, diagonal=1)

        ## Interacting
        # Select only interacting particles
        norms_interacting = norms_all_upper[self.interacting_idx][:, self.interacting_idx]
        n_within = torch.sum((norms_interacting > 0) & (norms_interacting < self.l_interacting))
        self.interaction_stats[0, self.t - 1] = n_within

        ## Non-interacting
        # Select only non-interacting particles
        norms_non_interacting = norms_all_upper[self.non_interacting_idx][:, self.non_interacting_idx]
        n_within = torch.sum((norms_non_interacting > 0) & (norms_non_interacting < self.l_interacting))
        self.interaction_stats[1, self.t - 1] = n_within

    def plot(self, x_plot, y_plot, z_plot, ax, label, ls='solid'):

        #all_condition = torch.zeros_like(self.states[0], dtype=bool)

        # Plot the different states
        for i in range(len(self.states)):
            ax.scatter(x_plot[self.states[i]].cpu(), y_plot[self.states[i]].cpu(), z_plot[self.states[i]].cpu(),
                       s=self.nucleosome_s, c=self.state_colors[i])
            #
            #all_condition = all_condition | self.states[i]

        # Plot chain line
        #all_condition = (self.state_A | self.state_B | self.state_C)
        all_condition = torch.ones_like(self.states[0], dtype=bool)

        ax.plot(x_plot[all_condition].cpu(), y_plot[all_condition].cpu(), z_plot[all_condition].cpu(),
                marker='o', ls=ls, markersize=self.chain_s, c='k', lw=0.7, label=label)

    def finalize_plot(self, ax):
        # Set 3D plot dimensions
        #plot_dim = (-1.5, 1.5)

        for i in range(len(self.state_colors)):
            ax.scatter([],[],c=self.state_colors[i],label=self.state_names[i])

        ax.legend(loc='upper left')
        ax.set(xlim=self.plot_dim, ylim=self.plot_dim, zlim=self.plot_dim)
        plt.show()

    def plot_statistics(self):
        s = 0.5
        fig,ax = plt.subplots(figsize=(8,6))
        ts = torch.arange(self.t_total)
        ax.scatter(ts, self.interaction_stats[0], s=s, label='Interacting states')
        ax.scatter(ts, self.interaction_stats[1], s=s, label='Non-interacting states')

        ax.set_xlabel(r'$t$', size=14)
        ax.set_ylabel('No. of interactions', size=14)
        plt.legend(loc='best')
        plt.show()

# Runs the script
def run(parameters):
    global t_total, sim_obj
    # Unpack parameters
    _, _, _, _, _, _, t_total, animate = parameters

    # Create simulation object
    sim_obj = Simulation(parameters)

    # Save initial state for plotting
    x_init = copy.deepcopy(sim_obj.X[:,0])
    y_init = copy.deepcopy(sim_obj.X[:,1])
    z_init = copy.deepcopy(sim_obj.X[:,2])

    coords_init = [x_init, y_init, z_init]

    # Simulation loop
    print('Simulation started.')

    if animate:
        # Create animation object
        anim_params = [sim_obj, t_total, coords_init]
        anim_obj = Animation(anim_params)

        # The animation loop
        # The function 'animation_loop' gets called t_total times
        anim = FuncAnimation(anim_obj.fig_anim, anim_obj.animation_loop, frames=anim_obj.frame_generator(),
                             interval=100, save_count=t_total+10)

        #filename = '/home/lars/Documents/masters_thesis/animation_no_polarity_100.gif'

        # Format
        writergif = animation.PillowWriter(fps=30)
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

        with torch.no_grad():
            sim_obj.plot(x_final, y_final, z_final, ax, label='Final state')

        # Plot MD
        sim_obj.finalize_plot(ax)
        # Plot statistics
        sim_obj.plot_statistics()

##############################################################################
##############################################################################