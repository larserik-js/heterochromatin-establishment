import numpy as np
import matplotlib.pyplot as plt


class System:

    def __init__(self):

        # Positions of monomers
        self.monomers = np.array([[-1, -1, -1],
                             [0, 0, 0],
                             [0.7, 1.2, 1.5],
                             [2, 2, 0.5],
                             [3, 3, 3]])

    def get_main_monomers(self):
        main_monomers = self.monomers[1:-1]

        return main_monomers

    def get_ghost_monomers(self):
        ghost_monomers = np.array([self.monomers[0],
                                   self.monomers[-1]])

        return ghost_monomers

    def get_all_coords(self):
        xs, ys, zs = self.monomers[:,0], self.monomers[:,1], self.monomers[:,2]

        return xs, ys, zs

    def get_main_coords(self):
        xs, ys, zs = self.get_all_coords()

        return xs[1:-1], ys[1:-1], zs[1:-1]

    def get_neighbor_coords(self):
        xs, ys, zs = self.get_all_coords()
        indices = [1,-2]

        neighbor_xs = xs[indices]
        neighbor_ys = ys[indices]
        neighbor_zs = zs[indices]

        return neighbor_xs, neighbor_ys, neighbor_zs

    def get_ghost_coords(self):
        ghost_monomers = self.get_ghost_monomers()

        return ghost_monomers[:,0], ghost_monomers[:,1], ghost_monomers[:,2]

    def get_dist_vect_between_neighbors(self):
        return self.monomers[3] - self.monomers[1]

    def get_point_between_neighbors(self):
        dist_vect_between_neighbors = self.get_dist_vect_between_neighbors()
        return self.monomers[1] + dist_vect_between_neighbors / 2

    def get_vectors(self):
        dist_vect_between_neighbors = self.get_dist_vect_between_neighbors()
        point_between_neighbors = self.get_point_between_neighbors()
        rot_vector = self.monomers[2] - point_between_neighbors
        dist_between_neighbors = np.linalg.norm(dist_vect_between_neighbors)

        rot_vector_ppdc = np.cross(dist_vect_between_neighbors, rot_vector)

        # Normalize
        rot_vector_ppdc /= dist_between_neighbors

        return rot_vector, rot_vector_ppdc

    def get_circle(self):
        rot_vector, rot_vector_ppdc = self.get_vectors()
        point_between_neighbors = self.get_point_between_neighbors()

        thetas = np.linspace(0,2*np.pi,1000, endpoint=False)

        circle = (point_between_neighbors
                  + np.cos(thetas)[:,None] * rot_vector
                  + np.sin(thetas)[:,None] * rot_vector_ppdc)

        return circle

    def get_dimensions(self):
        center_of_mass = self.monomers.sum(axis=0) / len(self.monomers)

        offset = 2

        x_dimension = (center_of_mass[0] - offset, center_of_mass[0] + offset)
        y_dimension = (center_of_mass[1] - offset, center_of_mass[1] + offset)
        z_dimension = (center_of_mass[2] - offset, center_of_mass[2] + offset)

        return x_dimension, y_dimension, z_dimension

def main():
    # Set up figure
    #fig = plt.figure(figsize=(4.792, 3.0))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Initialize instance
    system = System()
    monomers = system.monomers

    # Coordinates for monomers
    main_xs, main_ys, main_zs = system.get_main_coords()
    ghost_xs, ghost_ys, ghost_zs = system.get_ghost_coords()
    neighbor_xs, neighbor_ys, neighbor_zs = system.get_neighbor_coords()

    SCATTER_SIZE = 1000


    # Plot monomers
    ax.scatter(main_xs, main_ys, main_zs, s=SCATTER_SIZE, c='b', alpha=1)
    # Plot 'ghost' monomers
    ax.scatter(ghost_xs, ghost_ys, ghost_zs, s=SCATTER_SIZE, c='grey')

    # Chains between main monomers
    ax.plot(main_xs, main_ys, main_zs, c='k', alpha=1)
    # Chain from first ghost monomer
    ax.plot(monomers[:2,0], monomers[:2,1], monomers[:2,2],
            c='k', alpha=0.5)
    # Chain to last ghost monomer
    ax.plot(monomers[-2:,0], monomers[-2:,1], monomers[-2:,2],
            c='k', alpha=0.5)

    # Chain between end neighbor monomers
    ax.plot(neighbor_xs, neighbor_ys, neighbor_zs,
            c='k', ls='--', alpha=0.5)

    point_between_neighbors = system.get_point_between_neighbors()

    # Vectors
    rot_vector, rot_vector_ppdc = system.get_vectors()

    # ax.quiver(point_between_neighbors[0], point_between_neighbors[1],
    #           point_between_neighbors[2], rot_vector[0], rot_vector[1],
    #           rot_vector[2])
    #
    # ax.quiver(point_between_neighbors[0], point_between_neighbors[1],
    #           point_between_neighbors[2], rot_vector_ppdc[0], rot_vector_ppdc[1],
    #           rot_vector_ppdc[2])

    # Create rotation circle
    circle = system.get_circle()
    ax.scatter(circle[:,0], circle[:,1], circle[:,2],
               c='k', s=1, alpha=0.05)

    # Plot circle increment with arrow
    increment_idx = 150
    increment = circle[:increment_idx]
    ax.scatter(increment[:,0], increment[:,1], increment[:,2],
               c='brown', s=1, alpha=1)
    arrow = increment[-1] - increment[-2]
    ax.quiver(increment[-1,0], increment[-1,1], increment[-1,2],
              arrow[0], arrow[1], arrow[2], pivot='middle',
              arrow_length_ratio=20, color='brown')

    # Plot line from point between neighbors to monomer and arrow
    # To monomer
    ax.plot([point_between_neighbors[0], main_xs[1]],
            [point_between_neighbors[1], main_ys[1]],
            [point_between_neighbors[2], main_zs[1]],
            c='k', ls='--', alpha=0.5)

    # To arrow
    ax.plot([point_between_neighbors[0], circle[increment_idx,0]],
            [point_between_neighbors[1], circle[increment_idx,1]],
            [point_between_neighbors[2], circle[increment_idx,2]],
            c='k', ls='--', alpha=0.5)

    # Add text
    ax.text(0.2,0.7,0.7, r'$l_0$')
    ax.text(1.5,0.7,1.6, r'$l_0$')
    ax.text(0.92,1,1.07, r'$\frac{d \theta}{dt} \cdot \Delta t$', color='brown')

    # Set dimensions
    x_dimension, y_dimension, z_dimension = system.get_dimensions()

    ax.set(xlim=x_dimension, ylim=y_dimension, zlim=z_dimension)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    plt.show()


if __name__ == '__main__':
    main()