#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 21:27:28 2018

@author: cham
"""

#%%
""" download a grid of isochrones """
import numpy as np
from berliner.parsec import CMD
import joblib 
c = CMD()

# set grid
#grid_logt = [6, 7., 9]
#grid_feh = [-2.2, -1., 0, 1., 10]
#grid_lgage = np.arange(6, 10.15, 0.05)
#grid_mhini = np.arange(-2.9, 0.5, 0.1)

# get isochrones
grid_logage=(6, 10.2, 0.1)
grid_mh=(-2.6, 0.5, 0.1)

isoc_lgage, isoc_mhini, isoc_list_2mass_wise = c.get_isochrone_grid_mh(
    grid_logage=grid_logage, grid_mh=grid_mh, photsys_file="2mass_spitzer_wise",
    n_jobs=50, verbose=10)

isoc_lgage, isoc_mhini, isoc_list_sloan = c.get_isochrone_grid_mh(
    grid_logage=grid_logage, grid_mh=grid_mh, photsys_file="sloan",
    n_jobs=50, verbose=10)

isoc_lgage, isoc_mhini, isoc_list_ps1 = c.get_isochrone_grid_mh(
    grid_logage=grid_logage, grid_mh=grid_mh, photsys_file="panstarrs1", 
    n_jobs=50, verbose=10)

# combine photsys
for i in range(len(isoc_list_sloan)):
    for colname in isoc_list_ps1[i].colnames:
        if colname not in isoc_list_sloan[i].colnames:
            isoc_list_sloan[i].add_column(isoc_list_ps1[i][colname])
    for colname in isoc_list_2mass_wise[i].colnames:
        if colname not in isoc_list_sloan[i].colnames:
            isoc_list_sloan[i].add_column(isoc_list_2mass_wise[i][colname])
    print(i)

""" save data """
#joblib.dump((isoc_lgage, isoc_feh, isoc_list), 
#     "/home/cham/PycharmProjects/berliner/berliner/data/parsec12s_2mass_spitzer_wise.dump")

#joblib.dump((isoc_lgage, isoc_mhini, isoc_list), 
#     "/home/cham/PycharmProjects/berliner/berliner/data/parsec12s_2mass_spitzer_wise_hires.dump")

joblib.dump((isoc_lgage, isoc_mhini, isoc_list_sloan), 
     "/home/cham/PycharmProjects/berliner/berliner/data/cmd3.2_parsec12s_sloan_ps1_2mass_spitzer_wise.dump")

#%%
""" load data """
import numpy as np
import joblib
#isoc_lgage, isoc_feh, isoc_list = load(
#     "/home/cham/PycharmProjects/berliner/berliner/data/parsec12s_2mass_spitzer_wise.dump")

isoc_lgage, isoc_mhini, isoc_list = joblib.load(
     "/home/cham/PycharmProjects/berliner/berliner/data/cmd3.2_parsec12s_sloan_ps1_2mass_spitzer_wise.dump")

used_colnames = ['Zini', 'logAge', 'Mini', 'int_IMF', 'Mass', 'logL', 'logTe', 'logg', 'label', 'Mloss', 'X', 'Y', 'Xc', 'Xn', 'Xo', 'Cexcess', 'Z', 'mbolmag', 'umag', 'gmag', 'rmag', 'imag', 'zmag', 'gP1mag', 'rP1mag', 'iP1mag', 'zP1mag', 'yP1mag', 'wP1mag', 'Jmag', 'Hmag', 'Ksmag', 'W1mag', 'W2mag', 'W3mag', 'W4mag']
isoc_list = [_[used_colnames] for _ in isoc_list]

#%%
""" make an isochrone set """
from berliner import utils
ig = utils.IsochroneGrid(isoc_lgage, isoc_mhini, isoc_list, model="parsec")
print(ig)
print(np.sum(len(_) for _ in ig.isocs))

for i in range(len(ig.isocs)):
    ig.add_column(ig.isocs[i], 10**ig.isocs[i]["logTe"], "teff")
    

#%%

isoc0 = ig.get_a_isoc(9,0)
plot(isoc0["logTe"], isoc0["logg"], '-')
indplot = (isoc0["label"]>=1)&(isoc0["label"]<=8)
plot(isoc0["logTe"][indplot], isoc0["logg"][indplot], 'rs-')

""" interpolation """
igi = ig.isoc_linterp(restrictions=(('logg', 0.1), ('teff', 100)),
                      interp_colnames="all", Mini='Mini',
                      n_jobs=20, verbose=10)
print(ig)
print(igi)

""" make delta """
igi.make_delta()


#%%
""" given Teff logg [M/H], evaluate weighted mean of sed """
isocstack = igi.vstack()
isocstack.add_column((isocstack["_d_mini"]*isocstack["_d_mhini"]*isocstack["_d_age"]), name="w")
isocstack.write("/home/cham/PycharmProjects/berliner/berliner/data/cmd3.2_parsec12s_sloan_ps1_2mass_spitzer_wise.isocstack.fits")

from astropy import table
isocstack = table.Table.read("/home/cham/PycharmProjects/berliner/berliner/data/cmd3.2_parsec12s_sloan_ps1_2mass_spitzer_wise.isocstack.fits")
from berliner.utils import TGMMachine

tgm = TGMMachine(isocstack,tgm_cols=("teff", "logg", "_mhini"), tgm_sigma=(100, 0.2, 0.1), pred_cols=("teff", "logg", "_mhini"), wcol="w")

test_tgm_sun = np.array([[5778,4.3,0.0]])
test_tgm_kg = np.array([[4500,2.3,0.0]])
tgm.predict(test_tgm_kg)

tgm.predict(test_tgm_sun)

mod_sed = np.array(isocstack['umag', 'gmag', 'rmag', 'imag', 'zmag', 'gP1mag', 'rP1mag', 'iP1mag', 'zP1mag', 'yP1mag', 'wP1mag', 'Jmag', 'Hmag', 'Ksmag', 'W1mag', 'W2mag', 'W3mag', 'W4mag'].to_pandas())
mod_tgm = np.array(isocstack["teff", "logg", "_mhini"].to_pandas())
mod_mam = np.array(isocstack["Mini", "logAge", "_mhini"].to_pandas())


sigma_tgm = np.array([[100,0.2,0.1]])

test_tgm_sun = np.array([[5778,4.3,0.0]])
test_tgm_kg = np.array([[4500,2.3,0.0]])

#%%timeit
def eval_model(test_tgm):
    test_w = np.exp(-0.5*np.sum(((tgm-test_tgm)/sigma_tgm)**2., axis=1))*mod_w
    pred_sed = np.sum(mod_sed*test_w.reshape(-1,1), axis=0)/np.sum(test_w)
    #pred_mam = np.sum(mod_mam*test_w.reshape(-1,1), axis=0)/np.sum(test_w)
    return pred_sed

print(eval_model(test_tgm_sun))
print(eval_model(test_tgm_kg))

figure()
plot(eval_model(test_tgm_sun))
plot(eval_model(test_tgm_kg))
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

