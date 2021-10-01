import copy

import numpy as np
import torch
from numba import njit
import matplotlib.pyplot as plt

from scipy.special import lambertw, binom

class Simulation:
    def __init__(self, N, spring_strength, l0, noise, potential_weights, dt, t_total, classic):
        ## Polymer type
        self.classic = classic

        ## Parameters
        # No. of nucleosomes
        self.N = N
        # Half the spring constant
        self.spring_strength = spring_strength
        # Equilibrium spring length
        self.l0 = l0
        # Noise constant
        self.noise = noise
        # Potential weights
        self.potential_weights = potential_weights

        ## Initialize system
        thetas = np.linspace(0, 2*np.pi, N, endpoint=False)
        angle = thetas[1]-thetas[0]
        # The chord length
        r_system = self.l0 * np.sqrt( 1/(np.sin(angle)**2 + (1-np.cos(angle))**2 + (angle/np.pi)**2) )
        xs, ys = r_system*np.cos(thetas), r_system*np.sin(thetas)
        zs = np.linspace(-r_system,r_system,N)

        # Nucleosome positions
        self.X = torch.tensor([xs, ys, zs], dtype=torch.double).t()

        # No. of interacting nucleosomes
        self.n_interacting = int(N)
        # No. of allowed interactions
        self.n_allowed_interactions = 2

        # No. of possible interactions between interacting nucleosomes
        # The interaction between two particles is only counted once
        self.max_interactions = binom(self.n_interacting, 2)

        # States
        # Set no. of interacting nucleosomes to 0s
        # Set the rest randomly to 1s or 2s
        states = torch.zeros_like(self.X[:,0])
        states[self.n_interacting:] = torch.randint(1,3,(N-self.n_interacting,))
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

        # Particles within the following distance are counted for statistics
        self.l_interacting = 2*self.r0

        # Center of mass
        self.center_of_mass = torch.sum(self.X, dim=0) / self.N
        # Radius of gyration
        self.radius_of_gyration = 0

        ## Distance vectors from all particles to all particles
        #rij_all = self.X[:, None, :] - self.X
        rij_all = self.X - self.X[:, None, :]

        # Length of distances
        self.norms_all = torch.sqrt(1e-7 + torch.sum(rij_all**2, dim=2))  # torch.linalg.norm(rij_all, dim=2)

        # Normalized distance vectors
        self.rij_all = rij_all / (self.norms_all[:,:,None] + 1e-10)

        # Picks out nucleosomes that are allowed to interact with each other
        if self.classic:
            self.interaction_mask = torch.ones(size=(self.N, self.N), dtype=torch.bool)
            # Self-self interactions are forbidden
            self.interaction_mask.fill_diagonal_(False)
        else:
            self.interaction_mask = torch.from_numpy(self.get_interaction_mask())

        ## Polarity vectors
        # Set initial polarities
        self.P = torch.zeros_like(self.X)
        # Starting point: P_i = (0,0,1)
        self.P[:,2][self.interacting_idx] = 1

        ## Normalize polarity vectors
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
        move_to_gpu = False
        if move_to_gpu:
            self.X = self.X.cuda()
            self.mask = self.mask.cuda()

        # No. of time steps
        self.t = 0
        # Time-step
        self.dt = dt
        self.t_total = t_total
        self.t_half = int(t_total/2)

        ## Statistics
        # Will store the number of interactions that occur on a given neighbor-neighbor index difference
        self.interaction_idx_difference = torch.zeros(self.n_interacting, dtype=torch.float32)

        # Keeps track of the current lifetime of a given pairwise interaction
        self.running_lifetimes = torch.zeros(size=(self.n_interacting, self.n_interacting), dtype=torch.float)

        # Every time a life is completed, the lifetime is added to the relevant index
        self.lifetimes = torch.zeros_like(self.running_lifetimes, dtype=torch.float)

        # Number of completed lifetimes for a given interaction index difference
        # self.lifetimes divided by this number gives the average lifetime
        self.completed_lifetimes = torch.zeros_like(self.running_lifetimes, dtype=torch.float)

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

    # Norms between interacting particles only
    def get_interaction_norms(self):
        # Interactions are only counted once
        norms_all_upper = torch.triu(self.norms_all, diagonal=1)

        # Sliced matrices including interacting and non-interacting particles, respectively
        return norms_all_upper[self.interacting_idx][:, self.interacting_idx], \
               norms_all_upper[self.non_interacting_idx][:, self.non_interacting_idx]

    # Picks out nucleosomes that are allowed to interact with each other
    def get_interaction_mask(self):
        # Transform torch tensors to numpy array
        full_array = self.norms_all.detach().numpy()
        state_A = self.state_A.detach().numpy()

        # Indices for checking for possible interactions
        j_idx, i_idx = np.meshgrid(np.arange(self.N), np.arange(self.N))
        return self.mask_calculator(full_array, state_A, i_idx, j_idx, self.n_allowed_interactions)

    @staticmethod
    @njit
    def mask_calculator(full_array, state_A, i_idx, j_idx, n_allowed_interactions):
        # Total number of nucleosomes
        N = len(full_array)
        # Shows which nucleosomes interact with which
        interaction_mask = np.zeros(full_array.shape, dtype=np.bool_)

        # Sort distances
        sort_idx = np.argsort(full_array.flatten())
        i_idx = i_idx.flatten()[sort_idx]
        j_idx = j_idx.flatten()[sort_idx]

        # Number of allowed interactions
        #n_allowed_interactions = 2

        # Counts no. of interactions per nucleosome
        n_interactions = np.zeros(N, dtype=np.uint8)
        # For stopping criterion
        has_not_counted = np.ones(N, dtype=np.bool_)
        total_2_interactions = 0

        counter = 0

        # Loop over interaction distances
        for k in range(N, len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]

            # The interaction can only take place between state A nucleosomes
            not_interacting = not(state_A[i] and state_A[j])

            # If the nucleosomes are themselves or their nearest neighbors or not state A
            if i == j or i == j + 1 or i == j - 1\
                    or (interaction_mask[i, j] and interaction_mask[j, i]) or not_interacting:
                continue

            # If both nucleosomes are open for creating an interaction between them
            if n_interactions[i] < n_allowed_interactions and n_interactions[j] < n_allowed_interactions:
                counter += 1
                interaction_mask[i, j] = 1
                interaction_mask[j, i] = 1
                n_interactions[i] += 1
                n_interactions[j] += 1

                # For stopping criterion
                if has_not_counted[i] and n_interactions[i] == n_allowed_interactions:
                    total_2_interactions += 1
                    has_not_counted[i] = False
                if has_not_counted[j] and n_interactions[j] == n_allowed_interactions:
                    total_2_interactions += 1
                    has_not_counted[j] = False

            if total_2_interactions >= N - 1:
                #print('Loop ended at ' + str(counter))
                break

        # The returned mask has shape (N,N)
        return interaction_mask


    def grad_on(self):
        self.X.requires_grad_(True)
        #self.P.requires_grad_(True)

    def spring_potential(self):
        ## SPRING-BASED POTENTIAL
        # Distance vectors from a particle to its two neighbors in the chain
        rij = self.X[self.chain_idx] - self.X[:, None, :]
        # Length of the distance vectors
        norms = torch.sqrt(1e-5 + torch.sum(rij**2, dim=2))  # torch.linalg.norm(rij, dim=2)
        # Normalize distance vectors
        self.rij_hat = rij / (1e-10 + norms[:, :, None])
        # print(self.mask * (norms - self.l0)**2)
        # The spring-based potential term
        U_spring = self.spring_strength * torch.sum(self.mask * (norms - self.l0) ** 2)
        return U_spring

    # DISTANCE-BASED interaction potential
    def interaction_potential(self):
        # Regulate the potential function
        # b is the value which ensures that r0 is a local extremum for U_interaction
        b = np.real(-2 / lambertw(-2 * np.exp(-2)))

        U_interaction = torch.exp(-2 * self.norms_all / self.r0)

        # Leaves out self-self interactions
        mask_diag = (self.norms_all != 0)
        U_interaction = U_interaction * mask_diag

        U_attraction = -torch.exp(-2 * self.norms_all / (b * self.r0))

        U_interaction[self.interaction_mask] = U_interaction[self.interaction_mask] + U_attraction[self.interaction_mask]

        U_interaction = torch.sum(U_interaction)

        return U_interaction

    # # POLARITY-BASED interaction potential
    # def interaction_potential(self):
    #     # Regulate the potential function
    #     # b is the value which ensures that r0 is a local extremum for U_interaction
    #     b = np.real(-2 / lambertw(-2 * np.exp(-2)))
    #
    #     U_repulsion = torch.exp(-2 * self.norms_all / self.r0)
    #
    #     rij_attracting = self.rij_all[self.interacting_idx][:,self.interacting_idx]
    #
    #     # Row i contains the dot product of P_i with r_ij
    #     polar_attraction = torch.sum(self.P[self.interacting_idx][:,None,:] * rij_attracting, dim=2)
    #
    #     # Multiply clamped dot products for pair-wise interactions
    #     polar_attraction = torch.clamp(polar_attraction, min=0, max=1) * torch.clamp(polar_attraction, min=0, max=1).t()
    #
    #     # Normalize, such that the range of 'polar_attraction' becomes: [0,1]
    #     polar_attraction = torch.sum(polar_attraction /  (self.n_interacting**2 - self.n_interacting) )
    #
    #     if self.t % 1000 == 0:
    #         print(polar_attraction)
    #
    #     # Only calculate on interacting nucleosomes
    #     mask_attracting = self.state_A * self.state_A[:,None]
    #
    #     U_attraction = -polar_attraction * torch.exp(-2 * self.norms_all / (b * self.r0))
    #
    #     U_attraction = U_attraction * mask_attracting
    #
    #     # Leaves out self-self interactions
    #     mask_diag = (self.norms_all != 0)
    #
    #     U_interaction = torch.sum((U_repulsion + U_attraction)[mask_diag])
    #
    #     return U_interaction

    # Nuclear envelope pressure potential
    def pressure_potential(self):
        # Enacted by the nuclear envelope
        norms = torch.linalg.norm(self.X, dim=1)
        U_pressure = torch.sum(1/torch.abs(norms-4*self.r_system))
        #U_pressure = torch.sum(norms)
        return U_pressure

    # def twist_potential(self):
    #     neighbors = self.P[self.chain_idx]
    #     # The actual dot products
    #     dot_product = torch.sum(self.P[:,None,:] * neighbors, dim=2)
    #     # The total sum of the dot products
    #     dot_product = torch.sum(dot_product)
    #     # If all align, the potential is at its lowest
    #     return -dot_product

    # def p_directional_potential(self):
    #     neighbors = self.X[self.chain_idx]
    #
    #     # Pick out only interacting particles
    #     neighbors = neighbors[self.interacting_idx]
    #
    #     # For each particle, computes one distance vector between its neighbors
    #     r_between_neighbors = neighbors[:,1,:] - neighbors[:,0,:]
    #
    #     # Particles at the chain end: use the distance vector between this particle and its neighbor
    #     r_between_neighbors[0] = self.X[1] - self.X[0]
    #
    #     # Normalize
    #     r_between_neighbors = r_between_neighbors / (torch.linalg.norm(r_between_neighbors, dim=1)[:,None] + 1e-10)
    #
    #     dot_products = torch.sum(self.P[self.interacting_idx] * r_between_neighbors, dim=1)
    #
    #     # Potential is minimized when polarity vectors and r_between_neighbors are perpendicular
    #     U_p_directional = torch.sum(dot_products**2)
    #
    #     return U_p_directional

    # Returns (overall) system potential
    def potential(self):
        ## Spring-based potential
        U_spring = self.spring_potential()

        ## INTERACTION-BASED POTENTIAL
        U_interaction = self.interaction_potential()

        ## PRESSURE POTENTIAL
        U_pressure = self.pressure_potential()

        ## POLARITY VECTOR POTENTIALS
        #U_twist = self.twist_potential()

        ## This potential is minimized when the polarity vectors are perpendicular to the line between
        # the neighbors of its corresponding nucleosome
        #U_p_directional = self.p_directional_potential()

        ## Potential weights
        #w1, w2, w3, w4, w5 = self.potential_weights
        w1, w2, w3 = self.potential_weights

        #return w1 * U_spring + w2 * U_interaction + w3 * U_pressure + w4*U_twist + w5*U_p_directional
        return w1 * U_spring + w2 * U_interaction + w3 * U_pressure

    # Calculate radius of gyration
    def calculate_rg(self):
        distances_to_com = torch.norm(self.X - self.center_of_mass)
        return torch.sqrt( torch.mean(distances_to_com**2) )

    def gather_statistics(self):
        ## Equilibrium statistics are taken halfway through the simulation
        if self.t > self.t_half:

            ## Add new value of RG to itself
            # At the end divided by t_total for time average
            self.radius_of_gyration += self.calculate_rg()

            ## Count interaction distances
            # Interaction only applies to distances lower than l_interacting
            interaction_condition = (self.interaction_mask == True) & (self.norms_all < self.l_interacting)

            interaction_indices = torch.where(interaction_condition)
            interaction_distances = torch.abs((interaction_indices[1] - interaction_indices[0]))
            #print(f't = {self.t}')
            #print(interaction_distances)
            #print(torch.bincount(interaction_distances, minlength=self.N))
            # Only add a half as each distance is counted twice, but should only contribute once to statistics
            self.interaction_idx_difference += 0.5 * torch.bincount(interaction_distances, minlength=self.N)
            #print(self.interaction_idx_difference)

            ## Count lifetimes
            # If two nucleosomes are (still) interacting, add 1 to the running lifetimes
            self.running_lifetimes += interaction_condition.int()

            # If two nucleosomes are no longer interacting, reset the running lifetime, and count the reset

            reset_condition = (self.previous_interaction_mask & torch.logical_not(interaction_condition))

            self.lifetimes[reset_condition] += self.running_lifetimes[reset_condition]
            self.completed_lifetimes[reset_condition] += 1
            self.running_lifetimes[reset_condition] = 0

            ## Finalize statistics
            if self.t == self.t_total - 1:
                self.radius_of_gyration = self.radius_of_gyration / self.t_total

                self.average_lifetimes = torch.zeros(size=(self.N,), dtype=torch.float)

                for i in range(len(self.lifetimes)):
                    for k in range(len(self.lifetimes)-i-1):
                        j = k+i+1
                        idx = j-i
                        self.average_lifetimes[idx] += self.lifetimes[i,j] / (self.completed_lifetimes[i,j] + 1e-7)
                #print(self.interaction_idx_difference)

    def update(self):
        # Require gradient
        self.grad_on()

        ## Calculate potential
        U = self.potential()
        U.backward()

        ## Update variables
        with torch.no_grad():
            if torch.isnan(torch.sum(self.X.grad)):
                raise AssertionError('NAN in X gradient!')

            # Positions
            self.X -= self.X.grad * self.dt
            ## Polarities
            #self.P[self.interacting_idx] -= self.P.grad[self.interacting_idx] * self.dt

            # Add noise
            #self.X += self.noise * torch.randn_like(self.X) * np.exp(-self.t / self.t_total)
            #self.X += self.noise * torch.randn_like(self.X) * (1 - self.t/self.t_total)
            #self.X += self.noise * torch.randn_like(self.X)
            Du = 1
            eta = 1
            self.X += self.noise * torch.empty_like(self.X).normal_(mean=0, std=np.sqrt(2*Du/eta*self.dt))

            #self.P[self.interacting_idx] += self.noise * (torch.randn_like(self.X) * np.exp(-self.t / self.t_total))[self.interacting_idx]

            # Normalize polarity vectors
            #self.P = self.P / (torch.linalg.norm(self.P, dim=1)[:,None] + 1e-10)

            # Reset gradients
            self.X.grad.zero_()
            #self.P.grad.zero_()

        # New center of mass
        self.center_of_mass = torch.sum(self.X, dim=0) / self.N

        ## Distance vectors from all particles to all particles
        rij_all = self.X - self.X[:, None, :]

        # Copy previous interaction mask for statistics
        # This mask also includes the distance requirement for interactions
        self.previous_interaction_mask = copy.deepcopy(self.interaction_mask) & (self.norms_all < self.l_interacting)

        # Length of distances
        self.norms_all = torch.sqrt(1e-7 + torch.sum(rij_all**2, dim=2))  #  torch.linalg.norm(rij_all, dim=2)

        # Normalized distance vectors
        self.rij_all = rij_all / (self.norms_all[:,:,None] + 1e-10)

        # Create new interaction mask
        # This mask does NOT include the distance requirement for interactions
        if not self.classic:
            self.interaction_mask = torch.from_numpy(self.get_interaction_mask())

        # Count interactions for statistics
        with torch.no_grad():
            self.gather_statistics()

    def plot(self, x_plot, y_plot, z_plot, ax, label, ls='solid'):
        # Plot the different states
        for i in range(len(self.states)):
            ax.scatter(x_plot[self.states[i]].cpu(), y_plot[self.states[i]].cpu(), z_plot[self.states[i]].cpu(),
                       s=self.nucleosome_s, c=self.state_colors[i])

        # Plot chain line
        all_condition = torch.ones_like(self.states[0], dtype=torch.bool)

        ax.plot(x_plot[all_condition].cpu(), y_plot[all_condition].cpu(), z_plot[all_condition].cpu(),
                marker='o', ls=ls, markersize=self.chain_s, c='k', lw=0.7, label=label)

        # Plot center of mass
        ax.scatter(self.center_of_mass[0], self.center_of_mass[1], self.center_of_mass[2], s=0.5, c='g')

        # Plot polarity vectors
        #u, v, w = self.P[:,0], self.P[:,1], self.P[:,2]
        #ax.quiver(x_plot, y_plot, z_plot, u, v, w, length=1, normalize=True)

    def finalize_plot(self, ax):
        for i in range(len(self.state_colors)):
            ax.scatter([],[],c=self.state_colors[i],label=self.state_names[i])

        ax.legend(loc='upper left')
        ax.set_title(f'No. of nucleosomes = {self.N}', size=16)
        ax.set(xlim=self.plot_dim, ylim=self.plot_dim, zlim=self.plot_dim)
        plt.show()

    def plot_statistics(self):
        s = 0.5
        fig, ax = plt.subplots(2,1,figsize=(8,6))
        #ts = torch.arange(self.t_half)
        #ax[0].scatter(ts, self.interaction_stats[0], s=s, label='Interacting states')
        #ax[0].scatter(ts, self.interaction_stats[1], s=s, label='Non-interacting states')

        ax[1].plot(np.arange(self.n_interacting), self.interaction_idx_difference)

        ax[0].set_xlabel(r'$t$', size=14)
        ax[0].set_ylabel('No. of interactions', size=14)
        ax[0].set_title(f'No. of nucleosomes = {self.N}', size=16)

        ax[1].set_xlabel('Index difference', size=14)
        ax[1].set_ylabel('No. of interactions', size=14)

        #ax[0].legend(loc='best')
        plt.tight_layout()
        plt.show()
