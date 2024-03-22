"""
Test the performance of different methods in ``RegularGridInterpolator``.

method_list = ["linear", "nearest", "slinear", "cubic", "quintic", "pchip"]
"""

import numpy as np

x = np.linspace(0, 2, 6)
y = np.linspace(0, 2, 6)

xm, ym = np.meshgrid(x, y, indexing="ij")

xx = np.linspace(0, 2, 100)
yy = np.linspace(0, 2, 100)

xxm, yym = np.meshgrid(xx, yy, indexing="ij", sparse=False)

zm = np.random.randint(0, 5, size=xm.shape)


from scipy.interpolate import RegularGridInterpolator


def test_interp_method(method="linear"):
    rgi = RegularGridInterpolator(
        (x, y), zm, method=method, bounds_error=False, fill_value=-1
    )
    return rgi(np.vstack([xxm.flatten(), yym.flatten()]).T).reshape(xxm.shape)


import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


NROW = 3
NCOL = 3
npanel_interp = 4
fig = plt.figure(figsize=(10, 9))


ax = plt.subplot(NROW, NCOL, 1, projection="3d")
ax.plot_surface(xm, ym, zm, cmap=plt.cm.jet, label="truth")
ax.legend()
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlim(0, 5)
ax.view_init(elev=45, azim=45)

method_list = ["linear", "nearest", "slinear", "cubic", "quintic", "pchip"]
for i, method in enumerate(method_list):
    ax = plt.subplot(NROW, NCOL, i + npanel_interp, projection="3d")
    ax.plot_surface(
        xxm,
        yym,
        test_interp_method(method),
        cmap=plt.cm.jet,
        label=method,
        vmin=0,
        vmax=5,
    )
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlim(0, 5)
    ax.view_init(elev=45, azim=45)

fig.tight_layout()
