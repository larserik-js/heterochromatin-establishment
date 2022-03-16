import numpy as np

# Decaying sigmoid function
def sigmoid(x, shift):
    return 1 - 1 / (1 + np.exp(-(x + shift)))

# Constants are calculated in module 'fitpowerlaw.py'
def get_pressure(rms):
    SCALE = 186.76501549994606
    EXPONENT = -10.122352087102055
    SHIFT = -38.852017204731325

    return sigmoid(rms, SHIFT) * SCALE * rms**EXPONENT
