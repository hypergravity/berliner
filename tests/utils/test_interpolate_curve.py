import matplotlib.pyplot as plt
from berliner.interpolate.interpolate_curves import generate_3darc, interpolate_curve

"""
Test case 1: arcs
"""
x1 = generate_3darc(z=0, r=2, n_pnt=100)
x2 = generate_3darc(z=1, r=1, n_pnt=120)
x_weighted = interpolate_curve([x1, x1, x2], [1, 1, 1])

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot(*x1.T)
ax.plot(*x2.T)
ax.plot(*x_weighted.T)

xs = [x1, x2]
ws = [0.5, 0.5]
metric = 1.0
nsample = 1000

"""
Test case 2: tracks
"""
import joblib
import numpy as np

ts, ps = joblib.load(
    "/Users/cham/projects/starmod/mist_v1.2/eep_tracks/vvcrit0.0_tracks.joblib"
)
idx100 = np.where((ps[:, 0] == 1.00) & (ps[:, 1] == 0.00))[0][0]
idx102 = np.where((ps[:, 0] == 1.02) & (ps[:, 1] == 0.00))[0][0]
idx104 = np.where((ps[:, 0] == 1.04) & (ps[:, 1] == 0.00))[0][0]
print(idx100, idx102, idx104)
t100 = ts[idx100]
t102 = ts[idx102]
t104 = ts[idx104]

fig, axs = plt.subplots(2, 1, figsize=(10, 4))
ax = axs
ax.plot(np.log10(t100["star_age"]), t100["log_L"], label="1.00 Msun")
ax.plot(np.log10(t102["star_age"]), t102["log_L"], label="1.02 Msun")
ax.plot(np.log10(t104["star_age"]), t104["log_L"], label="1.04 Msun")

ax = axs[1]
ax.plot(t100["eep"], t100["log_L"], label="1.00 Msun")
ax.plot(t102["eep"], t102["log_L"], label="1.02 Msun")
ax.plot(t104["eep"], t104["log_L"], label="1.04 Msun")
