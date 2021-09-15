import matplotlib.pyplot as plt
import pickle
from mayavi import mlab

with open("/home/lars/Documents/masters_thesis/initial_final.pkl", "rb") as f:
    x, y, z, u, v, w = pickle.load(f)

mlab.plot3d(x, y, z)
mlab.quiver3d(x, y, z, u, v, w, mode="arrow", scale_factor=0.5)
mlab.show()
