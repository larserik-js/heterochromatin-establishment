import copy
import numpy as np
import torch
from numba import njit
import matplotlib.pyplot as plt
from scipy.special import lambertw
from statistics import _gather_statistics
from formatting import get_project_folder, create_param_string, create_plot_title
import pickle
r = np.random

class Simulation:
    def __init__(self, pathname, N, l0, noise, dt, t_total, U_two_interaction_weight, U_pressure_weight, alpha_1,
                 alpha_2, beta, stats_t_interval, seed, allow_state_change, initial_state, cell_division, cenH_size,
                 cenH_init_idx, write_cenH_data, barriers):

        # Project folder
        self.pathname = pathname

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

        # Total no. of time steps
        self.t_total = t_total
        self.t_half = int(self.t_total/2)

        # Time-step
        self.t = 0
        self.dt = dt

        # Potential weights
        self.U_two_interaction_weight = U_two_interaction_weight
        self.U_pressure_weight = U_pressure_weight

        ## State change parameters
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.beta = beta

        # Seed value
        self.seed = seed

        # Allow states to change
        self.allow_state_change = allow_state_change
        self.initial_state = initial_state

        # Include cell divisions
        self.cell_division = cell_division
        self.cell_division_interval = 2000000

        # Include cenH region
        self.cenH_size = cenH_size
        self.cenH_init_idx = cenH_init_idx
        self.cenH_indices = torch.arange(self.cenH_init_idx, self.cenH_init_idx + self.cenH_size)
        self.write_cenH_data = write_cenH_data

        # Include barriers
        self.barriers = barriers

        ## Initialize system
        self.X = self.initialize_system('quasi-random-pressure')

        # Half the chain length
        r_system = self.l0 * self.N / 2

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

        # All states
        self.states = self.initialize_states()

        # Pick out the nucleosomes of the different states
        self.state_S = (self.states==0)
        self.state_U = (self.states==1)
        self.state_A = (self.states==2)

        self.states_booleans = torch.cat([self.state_S[None,:], self.state_U[None,:], self.state_A[None,:]], dim=0)

        # Which type of interaction should be associated with the different states
        self.state_two_interaction = copy.deepcopy(self.state_S)
        self.state_unreactive = self.state_U | self.state_A

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

        # The interaction distance is set to half the equilibrium spring distance
        # The linker DNA in reality consists of up to about 80 bp
        self.r0 = self.l0 / 2

        # Particles within the following distance are counted for statistics
        self.l_interacting = 4 * self.r0

        # Regulate the potential function
        # b is the value which ensures that r0 is a local extremum for U_interaction
        self.b = np.real(-2 / lambertw(-2 * np.exp(-2)))

        # Cutoff distances for the potentials for the three different states
        self.potential_cutoff = 1*self.l0

        ## For statistics
        # Center of mass
        self.center_of_mass = torch.sum(self.X, dim=0) / self.N
        self.init_center_of_mass = torch.sum(self.X, dim=0) / self.N

        # All distance vectors from the nucleosomes to the center of mass
        self.dist_vecs_to_com = torch.empty(size=(int(self.t_total / stats_t_interval), self.N, 3))
        self.dist_vecs_to_com[0] = self.X - self.center_of_mass
        self.init_dist_vecs_to_com = self.dist_vecs_to_com[0]

        self.correlation_times = torch.zeros(size=(self.N,))

        # End-to-end distance
        self.Rs = torch.empty(size=(int(self.t_total / stats_t_interval),))
        self.end_to_end_vec_init = self.X[-1] - self.X[0]
        self.end_to_end_vec_dot = 999

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

        # The time when half the system is in an overall silent state
        self.half_silent_time = None
        # Number of silent state patches at the time half the system is in an overall silent state
        self.n_silent_patches = None
        # System in overall silent state
        self.stable_silent = False
        # Counts the number of particles in the different states
        self.state_statistics = torch.empty(size=(len(self.states_booleans), int(self.t_total / stats_t_interval)))

        self.states_time_space = torch.empty(size=(int(self.t_total / stats_t_interval), self.N))

        # Successful recruited conversions
        self.successful_recruited_conversions = torch.zeros(size=(4,self.N))
        self.successful_noisy_conversions = torch.zeros(size=(4,))

        ## Plot parameters
        self.plot_title = create_plot_title(self.U_pressure_weight, self.cenH_size, self.cenH_init_idx, self.barriers,
                                            self.N, self.t_total, self.noise, self.alpha_1, self.alpha_2, self.beta, self.seed)
        # Nucleosome scatter marker size
        self.nucleosome_s = 5
        # Chain scatter marker size
        self.chain_s = 1
        # Colors of scatter plot markers
        self.state_colors = ['r', 'y', 'b']
        self.state_names = ['Silent', 'Unmodified', 'Active']
        # Plot dimensions
        self.plot_dim = (-0.5*r_system, 0.5*r_system)
        self.r_system = r_system

        # File
        self.params_filename = create_param_string(self.U_pressure_weight, self.initial_state, self.cenH_size,
                                                   self.cenH_init_idx, self.cell_division, self.barriers, self.N, self.t_total,
                                                   self.noise, self.alpha_1, self.alpha_2, self.beta, self.seed)
        # Create figure
        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111, projection='3d')


    def initialize_system(self, init_system_type):
        # Quasi-random position based on position obtained after 1e6 time-steps of free polymer
        if init_system_type == 'quasi-random-free':
            seed_no = r.randint(100)
            open_filename = self.pathname + 'quasi_random_initial_states_free/'\
                                     + f'final_state_N=40_t_total=1000000_noise=0.500_seed={seed_no}.pkl'

            with open(open_filename, 'rb') as f:
                xs, ys, zs, _ = pickle.load(f)

        # With external pressure
        # Free polymer position after 1e6 time-steps
        # An additional 1e5 time-steps to let the pressure affect the polymer
        elif init_system_type == 'quasi-random-pressure':
            seed_no = r.randint(100)

            rounded_pressure_weight = np.round(self.U_pressure_weight, decimals=2)
            open_filename = self.pathname + 'quasi_random_initial_states_pressure_before_dynamics/'\
                                     + f'pressure={rounded_pressure_weight:.2f}/seed={seed_no}.pkl'

            with open(open_filename, 'rb') as f:
                X = pickle.load(f)

        # Stretched-out chain
        elif init_system_type == 'stretched':
            xs = torch.from_numpy(np.linspace(-(self.N - 1) / 2, (self.N - 1) / 2, self.N) * self.l0)
            ys, zs = torch.from_numpy(np.zeros(self.N)), torch.from_numpy(np.zeros(self.N))
            # Nucleosome positions
            X = torch.tensor([xs, ys, zs], dtype=torch.double).t()
        else:
            raise AssertionError("Invalid system type given in function 'initialize_system'!")

        return X

    def initialize_states(self):
        if self.initial_state == 'active':
            states = 2*torch.ones_like(self.X[:, 0], dtype=torch.int)
        elif self.initial_state == 'active_unmodified':
            states = 2*torch.ones_like(self.X[:, 0], dtype=torch.int)

            change_probs = torch.rand(size=(self.N,))
            change_conditions = change_probs >= 0.5

            # Change selected nucleosomes to unmodified
            states[change_conditions] = 1

        elif self.initial_state == 'unmodified':
            states = torch.ones_like(self.X[:, 0], dtype=torch.int)
        elif self.initial_state == 'unmodified_silent':
            states = torch.ones_like(self.X[:, 0], dtype=torch.int)

            change_probs = torch.rand(size=(self.N,))
            change_conditions = change_probs >= 0.5

            # Change selected nucleosomes to silent
            states[change_conditions] = 0

        elif self.initial_state == 'silent':
            states = torch.zeros_like(self.X[:, 0], dtype=torch.int)

        else:
            raise AssertionError('Invalid initial state given!')

        # Set cenH states to silent
        states[self.cenH_indices] = 0

        return states

    def states_after_cell_division(self):
        change_probs = torch.rand(size=(self.N,))
        change_conditions = change_probs >= 0.5

        # If cenH, do not change
        change_conditions[self.cenH_indices] = 0

        # Change selected nucleosomes to unmodified
        self.states[change_conditions] = 1

        return None

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

    # Set gradient to 0
    def grad_zero(self):
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
        # Enacted by the surroundings of the polymer
        norms = torch.linalg.norm(self.X - self.init_center_of_mass, dim=1)
        #U_pressure = torch.sum(1/(torch.abs(norms-2*self.r_system) + 1e-10) )

        # Hooke potential
        U_pressure = self.U_pressure_weight * torch.sum(norms**2)
        return U_pressure

    # Returns (overall) system potential
    def potential(self):
        ## INTERACTION-BASED POTENTIAL
        U_interaction = self.interaction_potential()

        ## PRESSURE POTENTIAL
        #U_pressure = 0
        U_pressure = self.pressure_potential()

        #return self.U_spring_weight * U_spring + U_interaction + self.U_pressure_weight * U_pressure
        #return U_interaction + self.U_pressure_weight * U_pressure
        return U_interaction + U_pressure


    # Uses imported function
    def gather_statistics(self):
        return _gather_statistics(self)

    @staticmethod
    @njit
    def _change_states(N, states, norms_all, l_interacting, alpha_1, alpha_2, beta, cenH_size, cenH_indices):

        # Particle on which to attempt a change
        n1_index = r.randint(N)

        # Does not change the cenH region
        if (cenH_size > 0) and (n1_index in cenH_indices):
            recruited_conversion_pair = None
            recruited_conversion_dist = None
            pass

        # If the nucleosome is not part of the cenH region
        else:
            # Recruited conversion
            # Other particles within distance
            particles_within_distance = \
            np.where((norms_all[n1_index] <= l_interacting) & (norms_all[n1_index] != 0))[0]

            # If there are other particles within l_interacting
            if len(particles_within_distance) > 0:

                # Choose one of those particles randomly
                n2_index = r.choice(particles_within_distance)

                # Do nothing if the recruiting nucleosome is unmodified, or
                # if the recruiting nucleosome is of the same state as the recruited nucleosome
                if states[n2_index] == 1 or states[n1_index] == states[n2_index]:
                    recruited_conversion_pair = None
                    recruited_conversion_dist = None
                    pass

                # Recruited conversion takes place
                else:
                    if states[n1_index] < states[n2_index]:
                        if r.rand() < alpha_2:
                            recruited_conversion_pair = (states[n1_index], states[n2_index])
                            states[n1_index] += 1
                        else:
                            recruited_conversion_pair = None
                            pass
                    elif states[n1_index] > states[n2_index]:
                        if r.rand() < alpha_1:
                            recruited_conversion_pair = (states[n1_index], states[n2_index])
                            states[n1_index] -= 1
                        else:
                            recruited_conversion_pair = None
                            pass
                    else:
                        raise AssertionError('Something is wrong in the change_states function!')

                    # The distance (in terms of indexed position in the chain) between the nucleosomes in the conversion
                    recruited_conversion_dist = np.abs(n1_index - n2_index)

            # No particles within l_interacting
            # No recruited conversion
            else:
                recruited_conversion_pair = None
                recruited_conversion_dist = None
                pass

        # Noisy conversion
        # Choose new random particle
        n1_index = r.randint(N)

        # Does not change the cenH region
        if (cenH_size > 0) and (n1_index in cenH_indices):
            noisy_conversion_idx = None
            pass

        # The chosen nucleosome is not in the cenH region
        else:
            if r.rand() < beta:
                # If the particle is in the S state
                if states[n1_index] == 0:
                    if r.rand() < alpha_2:
                        states[n1_index] = 1
                        noisy_conversion_idx = 0
                    else:
                        noisy_conversion_idx = None

                # If the particle is in the A state
                elif states[n1_index] == 2:
                    if r.rand() < alpha_1:
                        states[n1_index] = 1
                        noisy_conversion_idx = 2
                    else:
                        noisy_conversion_idx = None


                # If the particle is in the U state
                elif states[n1_index] == 1:
                    # Used to determine if the state should change to S or A
                    symmetric_rand = r.rand()

                    # U to A
                    if symmetric_rand < 0.5 and r.rand() < alpha_2:
                        states[n1_index] = 2
                        noisy_conversion_idx = 1

                    # U to S
                    elif symmetric_rand >= 0.5 and r.rand() < alpha_1:
                        states[n1_index] = 0
                        noisy_conversion_idx = 3
                    else:
                        noisy_conversion_idx = None
                else:
                    raise AssertionError("State other than 0, 1, or 2 given in function '_change_states'!")

            # No noisy conversion
            else:
                noisy_conversion_idx = None

        return states, recruited_conversion_pair, recruited_conversion_dist, noisy_conversion_idx


    # Changes the nucleosome states based on probability
    def change_states(self):
        # Numpy array
        states_numpy, recruited_conversion_pair, recruited_conversion_dist, noisy_conversion_idx = self._change_states(
                                        self.N, self.states.numpy(), self.norms_all.detach().numpy(),
                                        self.l_interacting, self.alpha_1, self.alpha_2, self.beta, self.cenH_size,
                                        self.cenH_indices.numpy())

        # Update the number of successful recruited conversions
        # S to U
        if recruited_conversion_pair == (0,2):
            self.successful_recruited_conversions[0,recruited_conversion_dist] += 1
        # U to A
        elif recruited_conversion_pair == (1,2):
            self.successful_recruited_conversions[1,recruited_conversion_dist] += 1
        # A to U
        elif recruited_conversion_pair == (2,0):
            self.successful_recruited_conversions[2,recruited_conversion_dist] += 1
        # U to S
        elif recruited_conversion_pair == (1,0):
            self.successful_recruited_conversions[3,recruited_conversion_dist] += 1
        # No recruited conversion
        elif recruited_conversion_pair == None:
            pass
        else:
            raise AssertionError("Invalid recruited conversion pair in function 'change_states'!")

        # Update the number of noisy conversions
        if noisy_conversion_idx == None:
            pass
        elif noisy_conversion_idx <= 3:
            self.successful_noisy_conversions[noisy_conversion_idx] += 1
        else:
            raise AssertionError(f"Invalid noisy conversion index: {noisy_conversion_idx}, type {type(noisy_conversion_idx)} in function 'change_states'!")

        # Change from Numpy array to Torch tensor
        states_torch = torch.from_numpy(states_numpy)
        self.states = copy.deepcopy(states_torch)

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
        if self.cell_division:
            if self.t % self.cell_division_interval == 0 and self.t != 0:
                # Initialize system in space
                self.X = self.initialize_system(init_system_type='quasi-random-free')
                self.X_tilde = self.get_X_tilde()
                # Change (on average) half the states to unmodified
                self.states_after_cell_division()

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
            with torch.no_grad():
                if index_type == 'even' or index_type == 'odd':
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

                # Reset gradients to 0
                self.grad_zero()

                # Reset angles to zero
                self.thetas = self.get_theta_zeros()
                self.X = self.X.detach()


        # Statistics
        with torch.no_grad():
            # New center of mass
            self.center_of_mass = torch.sum(self.X, dim=0) / self.N

            # Copy previous interaction mask for statistics
            # This mask also includes the distance requirement for interactions
            self.previous_interaction_mask = self.interaction_mask_two & (self.norms_all < self.l_interacting)

            # Update distance vectors
            self.rij_all, self.norms_all = self.get_norms()

            # Create new interaction mask
            # This mask does NOT include the distance requirement for interactions
            self.interaction_mask_two, self.interaction_mask_unreactive = self.get_interaction_mask()

            # Gather statistics
            self.gather_statistics()

            ## CHANGE STATES
            if self.allow_state_change:
                self.change_states()

                # Updates interaction types based on states
                self.update_interaction_types()

        # # Write pressure and RMS values
        # if self.t == self.t_total - 1:
        #     write_name = self.pathname + 'data/statistics/pressure_RMS_'
        #     write_name += f'init_state={self.initial_state}_cenH={self.cenH_size}_cenH_init_idx={self.cenH_init_idx}_N={self.N}_'\
        #                   f't_total={self.t_total}_noise={self.noise:.4f}_alpha_1={self.alpha_1:.5f}_alpha_2={self.alpha_2:.5f}_'\
        #                   f'beta={self.beta:.5f}_seed={self.seed}' + '.txt'
        #
        #     # Append to the file
        #     shape_0 = self.dist_vecs_to_com.shape[0]
        #     # RMS for different time steps
        #     RMS = torch.mean(torch.square(torch.norm(self.dist_vecs_to_com[int(shape_0/2):], dim=2)), dim=1)
        #
        #     # This mean is a time average from t_half to the end of the simulation
        #     mean_RMS = torch.mean(RMS)
        #     line_str = f'{self.U_pressure_weight},{mean_RMS:.4f}'
        #     data_file = open(write_name, 'a')
        #     data_file.write(line_str + '\n')
        #     data_file.close()

    # Clears the figure object and plots the current polymer
    def plot(self):
        # Clear the figure
        self.ax.clear()

        # Polymer position
        X, Y, Z = self.X[:,0].numpy(), self.X[:,1].numpy(), self.X[:,2].numpy()

        # Figure text
        text_str = r'$t = $' + f' {self.t} / {self.t_total}'
        r = self.r_system
        com = self.center_of_mass

        self.ax.text(r + com[0], -r + com[1], 1.8*r + com[2], text_str)

        # Plot chain
        self.ax.plot(X, Y, Z, lw=0.7, ls='solid', c='k')


        # Plot each state type separately
        for i in range(len(self.states_booleans)):
            x_plot = X[self.states_booleans[i]]
            y_plot = Y[self.states_booleans[i]]
            z_plot = Z[self.states_booleans[i]]

            self.ax.scatter(x_plot, y_plot, z_plot, s=self.nucleosome_s, c=self.state_colors[i],
                                 label=self.state_names[i])

        # Set plot dimensions
        self.ax.set(xlim=(com[0] + self.plot_dim[0], com[0] + self.plot_dim[1]),
               ylim=(com[1] + self.plot_dim[0], com[1] + self.plot_dim[1]),
               zlim=(com[2] + self.plot_dim[0], com[2] + self.plot_dim[1]))

        # Set title, labels and legend
        self.ax.set_title(self.plot_title, size=7)
        self.ax.set_xlabel('x', size=14)
        self.ax.set_ylabel('y', size=14)
        self.ax.set_zlabel('z', size=14)
        self.ax.legend(loc='upper left')

        return None


