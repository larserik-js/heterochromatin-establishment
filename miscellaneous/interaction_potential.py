import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw

r0 = 0.5
cutoff = 2*r0

rs = np.linspace(0,5,1000)
rs_below_cutoff = rs[rs < cutoff]


def U_interaction(r):
    b = np.real(-2 / lambertw(-2 * np.exp(-2)))
    return np.exp(-2*r / r0) - np.exp(-2*r / (b*r0))

fig, ax = plt.subplots(figsize=(4.792, 3.2))
#fig, ax = plt.subplots(figsize=(10,6))

# Plot potential function
Us = U_interaction(rs)
Us_below_cutoff = U_interaction(rs_below_cutoff)
U_min, U_max = Us_below_cutoff.min(), Us_below_cutoff.max()

ax.plot(rs, Us, c='r', ls='dotted', lw=2, alpha=1, label=r'$U_{interaction}$')
ax.plot(rs_below_cutoff, Us_below_cutoff, c='b', alpha=1, lw=0.95,
        label=r'$U_{interaction}$' + ', ' + r'$r < $' + ' Cutoff')

# Plot vline at r0
ax.vlines(cutoff, U_min + 0.1*U_min, U_max, color='k', ls='--')

# Set tick labels
ax.set_xticks([r0, cutoff])
ax.set_yticks([])

# Set xticklabels for the points
ax.set_xticklabels([r'$r_0$', 'Cutoff'])

# Format
ax.set_xlabel(r'$r$')
ax.set_ylabel(r'$U_{interaction}$')
ax.legend(loc='best')
fig.tight_layout()

plt.show()

