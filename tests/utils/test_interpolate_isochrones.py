"""
Test if it is possible to interpolate between isochrones.

Conclusion
----------
Negative.
"""

from berliner import CMD
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from berliner.interpolate.interpolate_curves import eval_angle_weighted_curve_length

c = CMD()


isoc8 = c.get_one_isochrone(logage=8.0, z=0.0152, mh=None, photsys_file="2mass_spitzer")
isoc802 = c.get_one_isochrone(
    logage=8.1, z=0.0152, mh=None, photsys_file="2mass_spitzer"
)
isoc805 = c.get_one_isochrone(
    logage=8.05, z=0.0152, mh=None, photsys_file="2mass_spitzer"
)


def plot(isoc, ax):
    x = np.array(isoc["Mini", "logTe", "logL"].to_pandas())
    cl = eval_angle_weighted_curve_length(
        x, metric=[1, 10, 10], cumulated=True, angle_weight_base=10
    )

    ax.plot(cl / cl[-1], isoc["logTe"], lw=2, label="logTe")
    ax.plot(cl / cl[-1], isoc["logg"], lw=2, label="logg")
    # ax.plot(cl / 100, lw=2, label="CL")


fig, ax = plt.subplots(1, 1, figsize=(15, 5))
plot(isoc8, ax)
plot(isoc802, ax)
ax.set_xlim(0, 1)
fig.tight_layout()


fig, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.plot(isoc8["logTe"], isoc8["logg"], lw=2, label="8.00")
ax.plot(isoc802["logTe"], isoc802["logg"], lw=2, label="8.02")
ax.plot(isoc805["logTe"], isoc805["logg"], lw=2, label="8.05")
ax.legend()

x1 = np.array(isoc8["Mini", "logTe", "logL"].to_pandas())
x2 = np.array(isoc802["Mini", "logTe", "logL"].to_pandas())
