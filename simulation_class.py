import copy
import numpy as np
import torch
from numba import njit
import matplotlib.pyplot as plt
from scipy.special import lambertw

from statistics import _gather_statistics

class Simulation:
    def __init__(self, N, spring_strength, l0, noise, U_spring_weight, U_two_interaction_weight,
                 U_classic_interaction_weight, U_pressure_weight, dt, t_total):

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
        self.U_spring_weight = U_spring_weight
        self.U_two_interaction_weight = U_two_interaction_weight
        self.U_classic_interaction_weight = U_classic_interaction_weight
        self.U_pressure_weight = U_pressure_weight


        ## Initialize system
        thetas = np.linspace(0, 2*np.pi, N, endpoint=False)
        angle = thetas[1]-thetas[0]
        # The chord length
        r_system = self.l0 * np.sqrt( 1/(np.sin(angle)**2 + (1-np.cos(angle))**2 + (angle/np.pi)**2) )
        xs, ys = r_system*np.cos(thetas), r_system*np.sin(thetas)
        zs = np.linspace(-r_system,r_system,N)

        # Nucleosome positions
        self.X = torch.tensor([xs, ys, zs], dtype=torch.double).t()

        # No. of allowed interactions in non-classic model
        self.n_allowed_interactions = 2

        # Mask to extract upper triangle
        self.mask_upper = torch.zeros(size=(self.N,self.N), dtype=torch.bool)
        self.triu_indices = torch.triu_indices(self.N, self.N, offset=1)
        self.mask_upper[self.triu_indices[0], self.triu_indices[1]] = 1

        ## States
        states = torch.zeros_like(self.X[:,0])
        # states[:10] = 0
        # states[10:20] = 2
        # states[20:30] = 0
        # states[30:40] = 2
        # states[40:50] = 0
        # states[50:60] = 2
        # states[60:70] = 0
        # states[70:80] = 2
        # states[80:90] = 0
        # states[90:] = 2
        states[:15] = 2
        states[15:30] = 0
        states[30:45] = 2
        states[45:60] = 0
        states[60:75] = 2
        states[75:] = 0
        #states = torch.ones_like(self.X[:,0])

        # Pick out the nucleosomes of the different states
        self.state_two_interaction = (states==0)
        self.state_classic = (states==1)
        self.state_unreactive = (states==2)
        self.states = [self.state_two_interaction, self.state_classic, self.state_unreactive]

        self.state_indices = [torch.where(self.state_two_interaction)[0],
                              torch.where(self.state_classic)[0],
                              torch.where(self.state_unreactive)[0]]

        ## Distance vectors from all particles to all particles
        rij_all = self.X - self.X[:, None, :]

        # Length of distances
        self.norms_all = torch.linalg.norm(rij_all, dim=2)

        # Normalized distance vectors
        self.rij_all = rij_all / (self.norms_all[:,:,None] + 1e-10)

        # Picks out nucleosomes that are allowed to interact with each other
        self.interaction_mask_two, self.interaction_mask_classic = self.get_interaction_mask()
        self.interaction_mask_two = torch.from_numpy(self.interaction_mask_two)
        self.interaction_mask_classic = torch.from_numpy(self.interaction_mask_classic)

        self.interaction_mask = self.interaction_mask_two + self.interaction_mask_classic

        # Diagonal indices
        self.diag_indices = self.norms_all != 0

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

        # Total no. of time steps
        self.t_total = t_total
        self.t_half = int(t_total/2)

        # Time-step
        self.t = 0
        self.dt = dt

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
        # For calculating interaction correlations
        self.shifts = np.arange(1,int(self.N/5),1)
        self.correlation_sums = np.zeros(len(self.shifts), dtype=float)

        # Will store the number of interactions that occur on a given neighbor-neighbor index difference
        self.interaction_idx_difference = torch.zeros(self.N, dtype=torch.float32)

        # Keeps track of the current lifetime of a given pairwise interaction
        self.running_lifetimes = torch.zeros(size=(self.N, self.N), dtype=torch.float)

        # Every time a life is completed, the lifetime is added to the relevant index
        self.lifetimes = torch.zeros_like(self.running_lifetimes, dtype=torch.float)

        # Number of completed lifetimes for a given interaction index difference
        # self.lifetimes divided by this number gives the average lifetime
        self.completed_lifetimes = torch.zeros_like(self.running_lifetimes, dtype=torch.float)
        self.average_lifetimes = torch.zeros(size=(self.N,), dtype=torch.float)

        ## Plot parameters
        # Nucleosome scatter marker size
        self.nucleosome_s = 5
        # Chain scatter marker size
        self.chain_s = 1
        # Colors of scatter plot markers
        self.state_colors = ['b', 'r', 'y']
        self.state_names = ['Two-interaction', 'Classic', 'Unreactive']
        # Plot dimensions
        self.plot_dim = (-1.5*r_system, 1.5*r_system)
        self.r_system = r_system

    # Picks out nucleosomes that are allowed to interact with each other
    def get_interaction_mask(self):
        # Transform torch tensors to numpy array
        norms_all = self.norms_all.detach().numpy()
        state_two_interaction = self.state_two_interaction.detach().numpy()
        state_classic = self.state_classic.detach().numpy()
        state_unreactive = self.state_unreactive.detach().numpy()

        # Indices for checking for possible interactions
        j_idx, i_idx = np.meshgrid(np.arange(self.N), np.arange(self.N))
        var1, var2 = self._mask_calculator(norms_all, state_two_interaction, state_classic, state_unreactive,
                                    i_idx, j_idx, self.n_allowed_interactions)
        return var1, var2

    @staticmethod
    @njit
    def _mask_calculator(norms_all, state_two_interaction, state_classic, state_unreactive,
                        i_idx, j_idx, n_allowed_interactions):
        # Total number of nucleosomes
        N = len(norms_all)
        # Shows which nucleosomes interact with which
        #interaction_mask = np.zeros(norms_all.shape, dtype=np.bool_)
        interaction_mask_two = np.zeros(norms_all.shape, dtype=np.bool_)
        interaction_mask_classic = np.zeros(norms_all.shape, dtype=np.bool_)

        # Sort distances
        sort_idx = np.argsort(norms_all.flatten())
        i_idx = i_idx.flatten()[sort_idx]
        j_idx = j_idx.flatten()[sort_idx]

        # Counts no. of interactions per nucleosome
        n_interactions = np.zeros(N, dtype=np.uint8)
        # For stopping criterion
        has_not_counted = np.ones(N, dtype=np.bool_)
        total_2_interactions = 0

        counter = 0

        # Loop over combinations of indices
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]

            # Checks if both nucleosomes are of the same (interacting) state
            two_interaction = state_two_interaction[i] and state_two_interaction[j]
            classic = state_classic[i] and state_classic[j]

            # Only nucleosomes of the same state can interact
            if not (two_interaction or classic):
                continue

            # If the nucleosomes in question are the same nucleosome
            # or if there already exists an interaction between them
            if i == j or (interaction_mask_two[i, j] and interaction_mask_two[j, i]) or\
                    (interaction_mask_classic[i, j] and interaction_mask_classic[j, i]):
                continue

            # If the nucleosomes are of the two-interaction state
            # and the nucleosomes in question are nearest neighbors
            if two_interaction and (i == j+1 or i == j-1):
                continue

            # Two-interaction state nucleosomes can only interact with max. 2 other nucleosomes
            if two_interaction and (n_interactions[i] >= n_allowed_interactions or n_interactions[j] >= n_allowed_interactions):
                continue

            # Create interaction
            counter += 1
            if two_interaction:
                # interaction_mask[i, j] = 1
                # interaction_mask[j, i] = 1
                interaction_mask_two[i,j] = 1
                interaction_mask_two[j,i] = 1
            elif classic:
                interaction_mask_classic[i,j] = 1
                interaction_mask_classic[j,i] = 1

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
        #return interaction_mask

        return interaction_mask_two, interaction_mask_classic

    def grad_on(self):
        self.X.requires_grad_(True)

    def spring_potential(self):
        ## SPRING-BASED POTENTIAL
        # Distance vectors from a particle to its two neighbors in the chain
        rij = self.X[self.chain_idx] - self.X[:, None, :]
        # Length of the distance vectors
        #norms = torch.sqrt(1e-5 + torch.sum(rij**2, dim=2))
        norms = torch.linalg.norm(rij, dim=2)
        # Normalize distance vectors
        self.rij_hat = rij / (1e-10 + norms[:, :, None])
        # The spring-based potential term
        U_spring = self.spring_strength * torch.sum(self.mask * (norms - self.l0) ** 2)
        return U_spring

    # DISTANCE-BASED interaction potential
    def interaction_potential(self):
        # Regulate the potential function
        # b is the value which ensures that r0 is a local extremum for U_interaction
        b = np.real(-2 / lambertw(-2 * np.exp(-2)) )

        ## Repulsion for all particles
        U_interaction = torch.exp(-2 * self.norms_all / self.r0)

        # Leaves out self-self interactions
        U_interaction = U_interaction * self.diag_indices

        ## Attraction potential
        # Only calculate on interacting nucleosomes within the interaction distance
        cutoff_interaction_mask = self.interaction_mask & (self.norms_all < self.l_interacting)
        U_attraction = -torch.exp(-2 * self.norms_all / (b * self.r0))

        # Add the attraction potential to the relevant particles
        U_interaction[cutoff_interaction_mask] = U_interaction[cutoff_interaction_mask] + U_attraction[cutoff_interaction_mask]

        # Multiply interactions by relevant potential weights
        U_interaction[self.interaction_mask_two] = U_interaction[self.interaction_mask_two] * self.U_two_interaction_weight

        U_interaction[self.interaction_mask_classic] = U_interaction[self.interaction_mask_classic] * self.U_classic_interaction_weight

        U_interaction[torch.logical_not(self.interaction_mask)] =\
            U_interaction[torch.logical_not(self.interaction_mask)] * self.U_two_interaction_weight

        return torch.sum(U_interaction)


    # Nuclear envelope pressure potential
    def pressure_potential(self):
        # Enacted by the nuclear envelope
        norms = torch.linalg.norm(self.X, dim=1)
        U_pressure = torch.sum(1/(torch.abs(norms-2*self.r_system) + 1e-10) )
        #U_pressure = torch.sum(norms)
        return U_pressure

    # Returns (overall) system potential
    def potential(self):
        ## Spring-based potential
        U_spring = self.spring_potential()

        ## INTERACTION-BASED POTENTIAL
        U_interaction = self.interaction_potential()

        ## PRESSURE POTENTIAL
        U_pressure = self.pressure_potential()

        return self.U_spring_weight * U_spring + U_interaction + self.U_pressure_weight * U_pressure

    def gather_statistics(self):
        return _gather_statistics(self)

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

            # Add noise
            #self.X += self.noise * torch.randn_like(self.X) * np.exp(-self.t / self.t_total)
            #self.X += self.noise * torch.randn_like(self.X) * (1 - self.t/self.t_total)
            #self.X += self.noise * torch.randn_like(self.X)
            Du = 1
            eta = 1
            self.X += self.noise * torch.empty_like(self.X).normal_(mean=0, std=np.sqrt(2*Du/eta*self.dt))

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
        self.norms_all = torch.linalg.norm(rij_all, dim=2)

        # Normalized distance vectors
        self.rij_all = rij_all / (self.norms_all[:,:,None] + 1e-10)

        # Create new interaction mask
        # This mask does NOT include the distance requirement for interactions
        self.interaction_mask_two, self.interaction_mask_classic = self.get_interaction_mask()
        self.interaction_mask_two = torch.from_numpy(self.interaction_mask_two)
        self.interaction_mask_classic = torch.from_numpy(self.interaction_mask_classic)

        self.interaction_mask = self.interaction_mask_two + self.interaction_mask_classic

        # Count interactions for statistics
        # Equilibrium statistics are taken halfway through the simulation
        if self.t > self.t_half:
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
