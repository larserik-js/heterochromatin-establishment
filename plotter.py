import matplotlib.pyplot as plt
import numpy as np
import pickle
from mayavi import mlab
from mayavi.mlab import *

with open("/home/lars/Documents/masters_thesis/final_state_N150.pkl", "rb") as f:
    x, y, z, u, v, w, states = pickle.load(f)

#mlab.plot3d(x, y, z)

colors = np.zeros(len(x), dtype=int)
print(colors)
exit()

for i in range(len(states)):
    colors[states[i]] = i

colors = colors[:,None]

scale_factor = 0.5
state_colors = ['b', 'r', 'y']
state_names = ['State A', 'State B', 'State C']



p3d = mlab.points3d(x,y,z, scale_factor=scale_factor)
p3d.module_manager.scalar_lut_manager.lut.number_of_colors = len(x)
p3d.module_manager.scalar_lut_manager.lut.table = colors

#mlab.quiver3d(x, y, z, u, v, w, mode="arrow", scale_factor=0.5)
mlab.draw()
mlab.show()
#
# def plot_matplotlib(x_plot, y_plot, z_plot, u, v, w):
#     ## Make figure
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Colors of scatter plot markers
#     state_colors = ['b', 'r', 'y']
#     state_names = ['State A', 'State B', 'State C']
#
#     # Plot the different states
#     for i in range(len(states)):
#         ax.scatter(x_plot[states[i]].cpu(), y_plot[states[i]].cpu(), z_plot[states[i]].cpu(),
#                    s=2, c=state_colors[i])
#
#     # Plot chain line
#     all_condition = torch.ones_like(states[0], dtype=bool)
#
#     ax.plot(x_plot[all_condition].cpu(), y_plot[all_condition].cpu(), z_plot[all_condition].cpu(),
#             marker='o', ls='-', markersize=0.1, c='k', lw=0.7, label='Test')
#
#     # Plot polarity vectors
#     #u, v, w = self.P[:,0], self.P[:,1], self.P[:,2]
#     ax.quiver(x_plot, y_plot, z_plot, u, v, w, length=1, normalize=True)
#
# plot_matplotlib(x,y,z,u,v,w,states)