import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

import get_pressure


class FitPowerLaw:

    def __init__(self):
        self.rms, self.pressure = self.get_data()
        self.initial_guess = np.array([1, -2, 0])

    #Read data from .txt file
    def get_data(self):
        pressure_rms_vals = np.loadtxt('pressure_RMS.txt', delimiter=',')
        pressure = pressure_rms_vals[:, 0]
        rms = pressure_rms_vals[:, 1]

        return rms[::-1], pressure[::-1]

    def cost_function(self, initial_guess):
        scale, exponent, shift = (initial_guess[0], initial_guess[1],
                                  initial_guess[2])

        return (get_pressure.sigmoid(self.rms, shift) * scale
                    * self.rms**exponent
                - self.pressure)

    def plot(self, scale, exponent, shift):
        fig, ax = plt.subplots(figsize=(4.792, 3))
        xs = np.linspace(self.rms[0], self.rms[-1], 1000)
        ys = get_pressure.sigmoid(xs, shift) * scale * xs ** exponent

        plt.plot(xs, ys, label='Fit function')
        plt.plot(self.rms, self.pressure, label='Experimental data')

        plt.legend(loc='best')
        plt.xlabel('RMS')
        plt.ylabel(r'$c_{pressure}$')

        fig.tight_layout()
        plt.show()

    def run(self):
        result = optimize.least_squares(self.cost_function, self.initial_guess)

        scale_fit = result.x[0]
        exponent_fit = result.x[1]
        shift_fit = result.x[2]

        print(f'Fit values: SCALE = {scale_fit:.3f}, '
              + f'EXPONENT = {exponent_fit:.3f}, SHIFT = {shift_fit:.3f}')

        self.plot(scale_fit, exponent_fit, shift_fit)


if __name__ == '__main__':
    fit_obj = FitPowerLaw()
    fit_obj.run()
