"""
Test downloading isochrones from CMD.
"""

# import CMD
from berliner import CMD
import matplotlib.pyplot as plt

# initialize it
c = CMD()

# Example 1: download one isochrone
isoc = c.get_one_isochrone(
    logage=7.0,  # log age
    z=0.0152,  # if [M/H] is not set, z is used
    mh=None,  # [M/H]
    photsys_file="2mass_spitzer",  # photometric system
)
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
ax.plot(isoc["logTe"], isoc["logg"], lw=2)
# ax.scatter(isoc9["logTe"], isoc9["logL"], s=5, c=isoc9["label"], cmap=plt.cm.jet)
for i in range(len(isoc)):
    ax.text(isoc["logTe"][i], isoc["logg"][i], isoc["label"][i])


# Example 2: download a grid of isochrones
# define your grid of logAge and [M/H] as tuple(lower, upper, step)
grid_logage = (6, 10.2, 0.1)
grid_mh = (-2.6, 0.5, 0.1)
# download isochrones in parallel
isoc_lgage, isoc_mhini, isoc_list_2mass_wise = c.get_isochrone_grid_mh(
    grid_logage=grid_logage,
    grid_mh=grid_mh,
    photsys_file="2mass_spitzer_wise",
    n_jobs=50,
    verbose=10,
)

# More ...
c.help()  # take a look at the output, it may help you!
