#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 21:27:28 2018

@author: cham
"""

#%%
""" download a grid of isochrones """
import numpy as np
from berliner.parsec import get_isochrone_grid
import joblib 

# set grid
#grid_logt = [6, 7., 9]
#grid_feh = [-2.2, -1., 0, 1., 10]
grid_logt = np.arange(6, 11, 0.05)
grid_feh = np.arange(-4, 1, 0.05)

# get isochrones
isoc_lgage, isoc_feh, isoc_list = get_isochrone_grid(
    grid_feh, grid_logt, model="parsec12s", phot="2mass_spitzer_wise", 
    n_jobs=30, verbose=10, silent=True)

""" save data """
#joblib.dump((isoc_lgage, isoc_feh, isoc_list), 
#     "/home/cham/PycharmProjects/berliner/berliner/data/parsec12s_2mass_spitzer_wise.dump")

joblib.dump((isoc_lgage, isoc_feh, isoc_list), 
     "/home/cham/PycharmProjects/berliner/berliner/data/parsec12s_2mass_spitzer_wise_hires.dump")

#%%
""" load data """
import numpy as np
import joblib
#isoc_lgage, isoc_feh, isoc_list = load(
#     "/home/cham/PycharmProjects/berliner/berliner/data/parsec12s_2mass_spitzer_wise.dump")

isoc_lgage, isoc_feh, isoc_list = joblib.load(
     "/home/cham/PycharmProjects/berliner/berliner/data/parsec12s_2mass_spitzer_wise_hires.dump")

#%%
""" make an isochrone set """
from berliner import utils
ig = utils.IsochroneGrid(isoc_lgage, isoc_feh, isoc_list, model="parsec12s")
print(ig)

#%%
""" interpolation """
igi = ig.isoc_linterp(restrictions=(('logG', 0.01), ('logTe', 0.01)),
                      interp_colnames="all", M_ini='M_ini',
                      n_jobs=20, verbose=10)
print(ig)
print(igi)

""" make delta """
igi.make_delta()

#%%
""" cut stages """
for i in range(len(igi.isocs)):
    ind = (igi.isocs[i]["stage"]>0) & (igi.isocs[i]["stage"]<9)
    igi.isocs[i] = igi.isocs[i][ind]
    igi.add_column(igi.isocs[i], 10.**igi.isocs[i]["logTe"], "teff")
    
#%%
""" 2d prior [teff, logg] """
qs, w = igi.make_prior_sb(func_prior=None, qs=["teff", "logG", "_fehini"])

bin_teff = np.linspace(3000., 15000., 200)
bin_logg = np.linspace(-1, 6, 100)

H, edges = np.histogramdd(qs[:, :2], 
                          bins=(bin_teff, bin_logg), 
                          density=True, weights=w)

joblib.dump((bin_teff, bin_logg, H), "/home/cham/PycharmProjects/berliner/berliner/data/prior_teff_logg.dump")

#%%
from skimage import filters
Hg = filters.gaussian(H, sigma=(1.))
joblib.dump((bin_teff, bin_logg, Hg), "/home/cham/PycharmProjects/berliner/berliner/data/prior_teff_logg_g1.0.dump")

#%%
""" make Regli instances """
centers = igi.edges_to_centers(edges)
meshs = np.meshgrid(*centers)
flats = np.array([_.flatten() for _ in meshs]).T
#from scipy.interpolate import RegularGridInterpolator
#RegularGridInterpolator
from regli import Regli
r = Regli.init_from_flats(flats)
r.set_values(np.log10(Hg).T.flatten())
r.redundant = True
joblib.dump(r, "/home/cham/PycharmProjects/berliner/berliner/data/regli_lnprior_teff_logg_g1.0.dump")

# check
#v_interp = r(flats[0])
#figure();
#imshow(np.log10(v_interp.reshape(*meshs[0].shape)))
#H.shape

#%%
""" make a plot of superposed prior """
# 1. comparison
#%pylab qt5
rcParams.update({"font.size":15})
fig1 = figure(figsize=(14, 6))
subplot(121)
imshow(np.log10(H.T), cmap=cm.jet, extent=(*bin_teff[[0, -1]], *bin_logg[[0, -1]]), aspect="auto", origin="lower")
colorbar()
xlim(*bin_teff[[-1,0]])
ylim(*bin_logg[[-1,0]])
xlabel("$T_{\\rm eff}$ [K]")
ylabel("$\log{g}$ [dex]")
title("superposed prior")

subplot(122)
imshow(np.log10(Hg).T, cmap=cm.jet, extent=(*bin_teff[[0, -1]], *bin_logg[[0, -1]]), aspect="auto", origin="lower")
colorbar()
xlim(*bin_teff[[-1,0]])
ylim(*bin_logg[[-1,0]])
xlabel("$T_{\\rm eff}$ [K]")
ylabel("$\log{g}$ [dex]")
title("superposed prior [smoothed (1.0)]")
fig1.tight_layout()
fig1.savefig("/home/cham/PycharmProjects/berliner/berliner/data/lnprior_teff_logg_comparison.svg")

# 2. single
fig2 = figure(figsize=(7, 6))
imshow(np.log10(Hg).T, cmap=cm.jet, extent=(*bin_teff[[0, -1]], *bin_logg[[0, -1]]), aspect="auto",  origin="lower")
colorbar()
xlim(*bin_teff[[-1,0]])
ylim(*bin_logg[[-1,0]])
xlabel("$T_{\\rm eff}$ [K]")
ylabel("$\log{g}$ [dex]")
title("superposed prior [smoothed (1.0)]")
fig2.tight_layout()
fig2.savefig("/home/cham/PycharmProjects/berliner/berliner/data/lnprior_teff_logg_smoothed1.0.svg")
fig2.savefig("/home/cham/PycharmProjects/berliner/berliner/data/lnprior_teff_logg_smoothed1.0.pdf")


#%%
#""" 3d prior [teff, logg, feh] """
#qs, w = igi.make_prior_sb(func_prior=None, qs=["teff", "logG", "_fehini"])
#
#n_feh_p_bin = 3
#d_fehini = np.unique(np.diff(igi.grid_fehini))
#
#bin_fehini = np.arange(igi.grid_fehini[0] - 0.5*d_fehini, igi.grid_fehini[-1] + (.5+n_feh_p_bin)*d_fehini, d_fehini*n_feh_p_bin)
#bin_teff = np.linspace(3000., 15000., 200)
#bin_logg = np.linspace(-1, 6, 100)
#
#H, edges = np.histogramdd(qs[:, :3], 
#                          bins=(bin_teff, bin_logg, bin_fehini), 
#                          density=True, weights=w)
#
#igi.edges_to_centers(edges)
#H.shape
#
##%%
#%pylab qt5
#figure()
#imshow(np.log10(H[:, :, 0].T), cmap=cm.jet)
#
#from skimage import filters
#Hgt = filters.gaussian(H, sigma=1.)
#imshow(np.log10(Hgt.T), cmap=cm.jet)
#colorbar()
#
##%%
#""" plot solar isochrone """
#isoc0 = ig.get_a_isoc(np.log10(5e9), 0)
#figure()
#plot(isoc0["logTe"], isoc0["logG"], 'o-')
#for i in range(len(isoc0)):
#    text(isoc0["logTe"][i], isoc0["logG"][i], "{}".format(isoc0["stage"][i]))

    