#import os
#os.environ['ETS_TOOLKIT'] = 'qt5'

import numpy as np
from mayavi import mlab
import pickle

print("da")

open_filename = "/home/lars/Documents/masters_thesis/final_state/final_state_N=150_noise=2,5.pkl"

with open(open_filename, 'rb') as f:
    x, y, z, u, v, w, states = pickle.load(f)


mlab.plot3d(x, y, z)
#
# colors = np.zeros(len(x), dtype=int)
#
#
# for i in range(len(states)):
#     colors[states[i]] = i
#
# colors = colors[:,None]
#
# scale_factor = 0.5
# state_colors = ['b', 'r', 'y']
# state_names = ['State A', 'State B', 'State C']
#
#
# p3d = mlab.points3d(x,y,z, scale_factor=scale_factor)
# p3 = mlab.plot3d(x,y,z)
# p3d.module_manager.scalar_lut_manager.lut.number_of_colors = len(x)
# p3d.module_manager.scalar_lut_manager.lut.table = colors

#mlab.quiver3d(x, y, z, u, v, w, mode="arrow", scale_factor=0.5)
mlab.draw()
mlab.show()