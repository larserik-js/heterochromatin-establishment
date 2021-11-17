import copy
import numpy as np
import torch
from numba import njit
import matplotlib.pyplot as plt
from scipy.special import lambertw

from statistics import _gather_statistics

r = np.random

class Simulation:
    def __init__(self, N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight, alpha_1, alpha_2, beta,
                 allow_state_change):

        ## Parameters
        # No. of nucleosomes
        self.N = N
        # N must be even
        if self.N % 2 != 0:
            raise AssertionError('N must be an even number!')

        # Equilibrium spring length
        self.l0 = l0
        # Noise constant
        self.noise = noise
        # Potential weights
        self.U_two_interaction_weight = U_two_interaction_weight
        self.U_pressure_weight = U_pressure_weight

        ## State change parameters
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.beta = beta

        # Allow states to change
        self.allow_state_change = allow_state_change

        ## Initialize system
        random_init = False
        if random_init:
            xs = [0.0]
            ys = [0.0]
            zs = [0.0]
            for i in range(self.N - 1):
                d = np.random.randn(3)
                d /= np.sqrt(np.sum(d**2))
                d *= self.l0
                xs.append(xs[-1] + d[0])
                ys.append(ys[-1] + d[1])
                zs.append(zs[-1] + d[2])
            xs = np.array(xs)
            ys = np.array(ys)
            zs = np.array(zs)

        else:
            xs = np.linspace(-(self.N - 1) / 2, (self.N - 1) / 2, self.N) * self.l0
            ys, zs = np.zeros(self.N), np.zeros(self.N)

        # Half the chain length
        r_system = self.l0 * self.N / 2

        # Nucleosome positions
        self.X = torch.tensor([xs, ys, zs], dtype=torch.double).t()

        # Index types to determine which nucleosomes to update
        self.index_types = ['even', 'odd', 'endpoints']

        # Vectors for picking out different indices
        m_even = torch.zeros(self.N)
        m_even[::2] = 1

        m_odd = torch.zeros(self.N)
        m_odd[1::2] = 1

        # Set endpoint indices to 0
        m_even[0], m_odd[-1] = 0, 0

        self.m_even, self.m_odd = m_even[:,None], m_odd[:,None]
        self.indexation_dict = {'even': m_even.bool(), 'odd': m_odd.bool(), 'endpoints': [0,-1]}

        # Points in the middle between the two neighboring particles
        self.X_tilde = self.get_X_tilde()

        # Rotation vectors
        self.rot_vector, self.rot_radius, self.rot_vector_ppdc = self.get_rot_vectors()

        # Angles for nucleosomes
        self.theta_zeros = self.get_theta_zeros()
        self.thetas = self.get_theta_zeros()

        # No. of allowed interactions in non-classic model
        self.n_allowed_interactions = 2

        # Mask to extract upper triangle
        self.mask_upper = torch.zeros(size=(self.N,self.N), dtype=torch.bool)
        self.triu_indices = torch.triu_indices(self.N, self.N, offset=1)
        self.mask_upper[self.triu_indices[0], self.triu_indices[1]] = 1

        ## States
        states = torch.zeros_like(self.X[:,0], dtype=torch.int)

        states[:int(self.N/2)] = 0
        states[int(self.N/2):] = 2

        #states = 2*torch.ones_like(self.X[:,0], dtype=torch.int)

        # Pick out the nucleosomes of the different states
        self.state_S = (states==0)
        self.state_U = (states==1)
        self.state_A = (states==2)

        # All states
        self.states = states

        self.states_booleans = torch.cat([self.state_S[None,:], self.state_U[None,:], self.state_A[None,:]], dim=0)

        # Which type of interaction should be associated with the different states
        self.state_two_interaction = copy.deepcopy(self.state_S)
        self.state_unreactive = self.state_U | self.state_A

        # ## TEST ##
        # self.state_two_interaction = (self.states==999)
        # self.state_unreactive = self.state_S | self.state_A

        ## Distance vectors from all particles to all particles
        self.rij_all, self.norms_all = self.get_norms()

        # (N,N) tensor with zeros along the diagonal and both off-diagonals
        self.wide_diag_zeros = torch.ones(size=(self.N, self.N), dtype=torch.double)
        for i in range(self.N):
            self.wide_diag_zeros[i, i] = 0
            if i > 0:
                self.wide_diag_zeros[i,i-1] = 0
            if i < self.N - 1:
                self.wide_diag_zeros[i,i+1] = 0

        # Same tensor with dtypes boolean
        self.wide_diag_zeros_bool = self.wide_diag_zeros.bool()

        # Picks out nucleosomes that are allowed to interact with each other
        self.interaction_mask_two, self.interaction_mask_unreactive = self.get_interaction_mask()

        # Total no. of time steps
        self.t_total = t_total
        self.t_half = int(t_total/2)

        # Time-step
        self.t = 0
        self.dt = dt

        # The interaction distance is set to half the equilibrium spring distance
        # The linker DNA in reality consists of up to about 80 bp
        self.r0 = self.l0 / 2

        # Particles within the following distance are counted for statistics
        self.l_interacting = 2 * self.r0

        # Regulate the potential function
        # b is the value which ensures that r0 is a local extremum for U_interaction
        self.b = np.real(-2 / lambertw(-2 * np.exp(-2)))

        # Cutoff distances for the potentials for the three different states
        self.potential_cutoff = 1*self.l0

        ## For statistics
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

        # Counts the number of particles in the different states
        self.state_statistics = torch.empty(size=(len(self.states_booleans), int(self.t_half / 25000)))

        # Measures distances from each nucleosome to the center of mass
        self.summed_distance_vecs_to_com = torch.zeros_like(self.X)
        self.distances_to_com = torch.empty(int(self.t_half / 25000))

        ## Plot parameters
        # Nucleosome scatter marker size
        self.nucleosome_s = 5
        # Chain scatter marker size
        self.chain_s = 1
        # Colors of scatter plot markers
        self.state_colors = ['b', 'r', 'y']
        self.state_names = ['Silent', 'Unmodified', 'Active']
        # Plot dimensions
        self.plot_dim = (-0.5*r_system, 0.5*r_system)
        self.r_system = r_system

    # Updates interaction types based on states
    def update_interaction_types(self):
        # Which type of interaction should be associated with the different states
        self.state_two_interaction = copy.deepcopy(self.state_S)
        self.state_unreactive = self.state_U | self.state_A

    # Picks out nucleosomes that are allowed to interact with each other
    def get_interaction_mask(self):
        # Transform torch tensors to numpy array
        norms_all = self.norms_all.detach().numpy()
        state_two_interaction = self.state_two_interaction.detach().numpy()

        # Indices for checking for possible interactions
        j_idx, i_idx = np.meshgrid(np.arange(self.N), np.arange(self.N))
        interaction_mask_two = self._mask_calculator(norms_all, state_two_interaction, i_idx, j_idx,
                                                     self.n_allowed_interactions)

        # Change from Numpy array to Torch tensor
        interaction_mask_two = torch.from_numpy(interaction_mask_two)

        # Construct the mask for unreactive states
        interaction_mask_unreactive = torch.logical_not(interaction_mask_two) & self.wide_diag_zeros_bool

        return interaction_mask_two, interaction_mask_unreactive

    @staticmethod
    @njit
    def _mask_calculator(norms_all, state_two_interaction, i_idx, j_idx, n_allowed_interactions):
        # Total number of nucleosomes
        N = len(norms_all)

        # Shows which nucleosomes interact with which
        interaction_mask_two = np.zeros(norms_all.shape, dtype=np.bool_)

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

            # Only S states can interact
            if not two_interaction:
                continue

            # If the nucleosomes are the same or nearest neighbors
            if i == j or i == j+1 or i == j-1:
                continue

            # If there already exists an interaction between the two nucleosomes
            if interaction_mask_two[i,j] and interaction_mask_two[j,i]:
                continue

            # Two-interaction state nucleosomes can only interact with max. 2 other nucleosomes
            if n_interactions[i] >= n_allowed_interactions or n_interactions[j] >= n_allowed_interactions:
                continue

            # Create interaction
            counter += 1

            interaction_mask_two[i, j] = 1
            interaction_mask_two[j, i] = 1

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

        return interaction_mask_two

    # Require gradient
    def grad_on(self):
        if self.index_type == 'even' or self.index_type == 'odd':
            self.thetas.requires_grad_(True)

        elif self.index_type == 'endpoints':
            self.X.requires_grad_(True)
        else:
            raise AssertionError('Invalid index type given in function "update".')

    # Turn off gradient
    def grad_off(self):
        if self.index_type == 'even' or self.index_type == 'odd':
            # Reset gradients
            self.thetas.grad.zero_()

        elif self.index_type == 'endpoints':
            self.X.grad.zero_()
        else:
            raise AssertionError('Invalid index type given in function "update".')

    # DISTANCE-BASED interaction potential
    def interaction_potential(self):
        # Only calculate on interacting nucleosomes within the interaction distance

        mask_two_cutoff = (self.interaction_mask_two & (self.norms_all < self.potential_cutoff)).double()
        mask_all_cutoff = (self.wide_diag_zeros_bool & (self.norms_all < self.potential_cutoff)).double()

        U_interaction = torch.exp(-2 * self.norms_all / self.r0) * mask_all_cutoff
        U_interaction = U_interaction - torch.exp(-2 * self.norms_all / (self.b * self.r0)) * mask_two_cutoff

        return self.U_two_interaction_weight * torch.sum(U_interaction)

    # Nuclear envelope pressure potential
    def pressure_potential(self):
        # Enacted by the nuclear envelope
        norms = torch.linalg.norm(self.X, dim=1)
        U_pressure = torch.sum(1/(torch.abs(norms-2*self.r_system) + 1e-10) )
        #U_pressure = torch.sum(norms)
        return U_pressure

    # Returns (overall) system potential
    def potential(self):
        ## INTERACTION-BASED POTENTIAL
        U_interaction = self.interaction_potential()

        ## PRESSURE POTENTIAL
        U_pressure = self.pressure_potential()

        #return self.U_spring_weight * U_spring + U_interaction + self.U_pressure_weight * U_pressure
        #return U_interaction + self.U_pressure_weight * U_pressure
        return U_interaction


    # Uses imported function
    def gather_statistics(self):
        return _gather_statistics(self)

    @staticmethod
    @njit
    def _change_states(N, states, norms_all, l_interacting, alpha_1, alpha_2, beta):

        # Particle on which to attempt a change
        n1_index = r.randint(N)

        # Choose reaction probability based on the state of n1
        if states[n1_index] == 0:
            alpha = alpha_2 + 0
        elif states[n1_index] == 2:
            alpha = alpha_1 + 0
        elif states[n1_index] == 1:
            alpha = (alpha_1 + alpha_2) / 2
        else:
            raise AssertionError('State not equal to 0, 1, or 2.')

        # Recruited conversion
        rand_alpha = r.rand()

        if rand_alpha < alpha:

            # Other particles within distance
            particles_within_distance = \
            np.where((norms_all[n1_index] <= l_interacting) & (norms_all[n1_index] != 0))[0]

            # If there are other particles within l_interacting
            if len(particles_within_distance) > 0:

                # Choose one of those particles randomly
                n2_index = r.choice(particles_within_distance)

                # If the n2 state is U, do not perform any changes
                if states[n1_index] < states[n2_index] and states[n2_index] != 1:
                    states[n1_index] += 1
                elif states[n1_index] > states[n2_index] and states[n2_index] != 1:
                    states[n1_index] -= 1

        # Noisy conversion
        # Choose new random particle
        n1_index = r.randint(N)

        rand_beta = r.rand()
        if rand_beta < beta:

            if states[n1_index] == 0:
                states[n1_index] += 1
            elif states[n1_index] == 2:
                states[n1_index] -= 1

            else:
                # If the particle is in the U state, choose a change to A or S randomly
                rand = r.rand()
                if states[n1_index] == 1 and rand < 0.5:
                    states[n1_index] += 1
                elif states[n1_index] == 1 and rand >= 0.5:
                    states[n1_index] -= 1

        return states

    # Changes the nucleosome states based on probability
    def change_states(self):
        self.states = self._change_states(self.N, self.states.numpy(), self.norms_all.detach().numpy(),
                                          self.l_interacting, self.alpha_1, self.alpha_2, self.beta)

        # Change from Numpy array to Torch tensor
        self.states = torch.from_numpy(self.states)

        # Update individual (boolean) tensors
        self.state_S, self.state_U, self.state_A = (self.states==0), (self.states==1), (self.states==2)
        self.states_booleans = torch.cat([self.state_S[None,:], self.state_U[None,:], self.state_A[None,:]], dim=0)

        return None

    # Returns a tensor of zeros
    def get_theta_zeros(self):
        return torch.zeros(self.N)[:, None]

    # For each nucleosome except endpoint nucleosomes
    # The distance between its two neighbors
    def get_d_between_neighbors(self):
        d_between_neighbors = torch.zeros_like(self.X)
        d_between_neighbors[1:-1] = self.X[2:] - self.X[:-2]
        return d_between_neighbors

    # For each nucleosome except endpoint nucleosomes
    # The point exactly between its two neighbors
    def get_X_tilde(self):
        d_between_neighbors = self.get_d_between_neighbors()
        X_tilde = torch.zeros_like(self.X)
        X_tilde[1:-1] += self.X[:-2] + d_between_neighbors[1:-1] / 2
        return X_tilde

    # Returns position-dependent vectors
    def get_rot_vectors(self):
        d_between_neighbors = self.get_d_between_neighbors()

        # Rotation vector for each particle
        rot_vector = self.X - self.X_tilde
        # Radius of rotation
        rot_radius = torch.norm(rot_vector, dim=1)
        # Perpendicular vector (normalized to be distance rot_radius)
        rot_vector_ppdc = torch.cross(rot_vector, d_between_neighbors)
        rot_vector_ppdc /= torch.norm(d_between_neighbors + 1e-10, dim=1)[:, None]

        return rot_vector, rot_radius, rot_vector_ppdc

    # Returns the angle-dependent position tensor
    def get_X_theta(self):
        tilde_plus_angles = self.X_tilde + torch.cos(self.thetas) * self.rot_vector \
                            + torch.sin(self.thetas) * self.rot_vector_ppdc

        if self.index_type == 'odd':
            X_theta = self.m_even * self.X + self.m_odd * tilde_plus_angles

        elif self.index_type == 'even':
            X_theta = self.m_odd * self.X + self.m_even * tilde_plus_angles

        else:
            raise AssertionError('Invalid index type found in function "get_X_theta".')

        X_theta[0] += self.X[0]
        X_theta[-1] += self.X[-1]

        return X_theta

    # For all nucleosomes
    # The normalized distance vectors to all other nucleosomes
    # As well as the length of these distances
    def get_norms(self):
        # Distance vectors from all particles to all particles
        rij_all = self.X - self.X[:, None, :]

        # Length of distances
        norms_all = torch.linalg.norm(rij_all, dim=2)

        # Normalized distance vectors
        rij_all = rij_all / (norms_all[:,:,None] + 1e-10)

        return rij_all, norms_all

    # The update step
    def update(self):
        # For update of nucleosomes of even and odd indices, plus the nucleosomes at each end of the chain
        for index_type in self.index_types:
            # Set index type for the update
            self.index_type = index_type

            # Boolean tensor for indexing
            indexation = self.indexation_dict[index_type]

            # Requires gradient for:
            # Thetas if indices even or odd, or:
            # Xs if indices are the endpoints
            self.grad_on()

            # Get position matrix
            if index_type == 'even' or index_type == 'odd':
                self.X = self.get_X_theta()

            self.rij_all, self.norms_all = self.get_norms()

            ## Calculate potential
            U = self.potential()
            U.backward()

            ## Update variables
            if index_type == 'even' or index_type == 'odd':

                with torch.no_grad():
                    if torch.isnan(torch.sum(self.thetas.grad)):
                        raise AssertionError('NAN in gradient!')

                    # Update angles for non-endpoints
                    self.thetas[indexation] -= self.thetas.grad[indexation] * self.dt

                    # Add noise
                    #self.X += self.noise * torch.randn_like(self.X) * np.exp(-self.t / self.t_total)
                    #self.X += self.noise * torch.randn_like(self.X) * (1 - self.t/self.t_total)
                    #self.X += self.noise * torch.randn_like(self.X)
                    Du = 1
                    eta = 1

                    self.thetas[indexation] += self.noise * torch.empty_like(self.thetas[indexation]).\
                        normal_(mean=0, std=np.sqrt(2 * Du / eta * self.dt)) / (self.rot_radius[indexation][:,None] + 1e-10)

                    # Update positions and vectors
                    self.X = self.get_X_theta()

            elif index_type == 'endpoints':
                with torch.no_grad():
                    if torch.isnan(torch.sum(self.X.grad)):
                        raise AssertionError('NAN in gradient!')

                    # Update positions for endpoints
                    self.X[indexation] -= self.X.grad[indexation] * self.dt

                    # Add noise
                    Du = 1
                    eta = 1

                    self.X[indexation] += self.noise * torch.empty_like(self.X[indexation]).normal_(mean=0, std=np.sqrt(2 * Du / eta * self.dt))

                    # Adjust distances from endpoints to their neighbors back to l0
                    endpoint_d_vecs = self.X[indexation] - self.X[[1,-2]]
                    self.X[indexation] = self.X[[1,-2]] + endpoint_d_vecs * (self.l0 / torch.linalg.norm(endpoint_d_vecs, dim=1)[:,None])

            self.X_tilde = self.get_X_tilde()
            self.rot_vector, self.rot_radius, self.rot_vector_ppdc = self.get_rot_vectors()

            # Reset gradients
            self.grad_off()

            # Reset angles to zero
            self.thetas = self.get_theta_zeros()

        # New center of mass
        self.center_of_mass = torch.sum(self.X, dim=0) / self.N

        # if self.t == self.t_total - 1:
        #     fig_stats, ax_stats = plt.subplots()
        #     ax_stats.plot(np.arange(len(self.distances_to_com)), self.distances_to_com.numpy())
        #     plt.show()

        # Copy previous interaction mask for statistics
        # This mask also includes the distance requirement for interactions
        self.previous_interaction_mask = copy.deepcopy(self.interaction_mask_two) & (self.norms_all < self.l_interacting)

        # Update distance vectors
        self.rij_all, self.norms_all = self.get_norms()

        # Create new interaction mask
        # This mask does NOT include the distance requirement for interactions
        self.interaction_mask_two, self.interaction_mask_unreactive = self.get_interaction_mask()

        # Count interactions for statistics
        # Equilibrium statistics are taken halfway through the simulation
        if self.t >= self.t_half:
            with torch.no_grad():
                self.gather_statistics()

        ## CHANGE STATES
        if self.allow_state_change:
            self.change_states()

            # Updates interaction types based on states
            self.update_interaction_types()


    def plot(self, x_plot, y_plot, z_plot, ax, label, ls='solid'):
        # Plot the different states
        for i in range(len(self.states_booleans)):
            ax.scatter(x_plot[self.states_booleans[i]].cpu(), y_plot[self.states_booleans[i]].cpu(),
                       z_plot[self.states_booleans[i]].cpu(), s=self.nucleosome_s, c=self.state_colors[i])

        # Plot chain line
        all_condition = torch.ones_like(self.states_booleans[0], dtype=torch.bool)

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
