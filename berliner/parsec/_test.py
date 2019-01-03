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

t.post_proc("parsec12s")
t.pprint()
print(t)
t.show_in_browser()
#
Isochrone(t.columns)

#%%
""" download a grid of isochrones """
import numpy as np
from berliner.parsec.isochrone_grid import get_isochrone_grid

# set grid
grid_logt = [6, 7., 9]
grid_feh = [-2.2, -1., 0, 1., 10]

grid_logt = np.arange(6, 11, 0.2)
grid_feh = np.arange(-4, 1, 0.2)

# get isochrones
isoc_lgage, isoc_feh, isoc_list = get_isochrone_grid(
    grid_feh, grid_logt, model="parsec12s", phot="2mass_spitzer_wise", n_jobs=20, verbose=10, silent=True)
