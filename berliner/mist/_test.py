# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 23:36:54 2018

@author: cham
"""

#%%
from berliner import mist
import glob
data_dir = "/hydrogen/mist/1.2/isochrones/MIST_v1.2_vvcrit0.4_WISE"
fps = glob.glob(data_dir+"/*.cmd")
print(fps)

filepath = fps[0]
isocs = mist.read_mist_isochrones(fps[0])
isoc0 = isocs[0]

from matplotlib import pyplot as plt
plt.figure()
plt.plot(isoc0["log_Teff"], isoc0["log_g"])

#%%
""" read all isochrones from pattern """
from berliner import mist
ptn = "/hydrogen/mist/1.2/isochrones/MIST_v1.2_vvcrit0.4_WISE/*.cmd"
isoclist = mist.read_isochrones_ptn(ptn)


#%%
""" read all tracks from pattern """
#import glob
#fps = glob.glob(ptn)
from berliner import mist, utils
ptn = "/hydrogen/mist/1.2/eep_tracks/MIST_v1.2_*_vvcrit0.0_EEPS/*.track.eep"
tracks, metatable = mist.read_tracks_ptn(ptn)

import numpy as np
ts = utils.TrackSet(np.log10(metatable["initial_mass"]), metatable["FeH"], tracks)

ts.interp_tgm([1.,0.01,209])
