# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 13:33:05 2018

@author: cham
"""

#%%
from berliner import parsec
from berliner.parsec import isochrone_grid, Isochrone
#isochrone_grid._test()

t = parsec.get_one_isochrone_silently(1e9, 0.0152)
t = parsec.get_one_isochrone(1e9, 0.0152)

t.post_proc("parsec")
print(t)
t.show_in_browser()
#
Isochrone(t.columns)
#%%
from berliner.parsec.isochrone_grid import get_isochrone_grid
    (get_isochrone_grid, interpolate_to_cube, cubelist_to_hdulist,
     combine_isochrones, write_isoc_list)

# set grid
grid_logt = [6, 7., 9]
grid_feh = [-2.2, -1., 0, 1., 10]
grid_mini = np.arange(0.01, 12, 0.01)

# get isochrones
vgrid_feh, vgrid_logt, grid_list, isoc_list = get_isochrone_grid(
    grid_feh, grid_logt, model="parsec12s", phot="sloan", n_jobs=1)