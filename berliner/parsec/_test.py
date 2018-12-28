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

#%%
""" isochrone set """
from berliner import utils
isoset = utils.IsochroneSet(isoc_lgage, isoc_feh, isoc_list)
isoset.isocs
print(isoset.colnames)

isoset.post_proc(model="parsec")
print(isoset.colnames)
print(isoset[0])
#isoset[0].show_in_browser()
isoset.vstack()
#%%
""" interpolation """

isoset_interp = isoset.isoc_linterp(restrictions=(('logG', 0.01), ('logTe', 0.01)), interp_colnames="all", M_ini='M_ini', n_jobs=1)

print(isoset)
print(isoset_interp)

#%%
""" make a prior """

#%%
# %pylab qt5
from matplotlib.pyplot import *
figure()
for isoc in isoset.isocs:
    plot(isoc["logTe"], isoc["logG"])
    
    
#%%
from ezpadova import get_one_isochrone
isoc0 = get_one_isochrone(1E9, 0.0152, model="parsec12s", phot="sloan",)
isoc1 = get_one_isochrone(1E9, 0.00152)


from astropy.table import Table, vstack
isoc0_ = Table(isoc0.data)
isoc1_ = Table(isoc1.data)
vstack([isoc0_, isoc1_,])

    