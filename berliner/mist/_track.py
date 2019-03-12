import glob
import re
from collections import OrderedDict
import copy

import numpy as np
from astropy.table import Table, Column
from joblib import Parallel, delayed

Zsun = 0.0142857


def read_tracks_ptn(ptn="./*.track.eep", colnames=None, n_jobs=-1, verbose=5,
                    metatable=True):
    """ read multiple MIST track files """
    fps = glob.glob(ptn)
    print("@mist: reading tracks ...")
    track_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(read_track)(fp, colnames) for fp in fps)

    meta0 = copy.deepcopy(track_list[0].meta)
    meta0.pop("EEPs")
    names = tuple(meta0.keys())

    meta_data_rows = []
    for track in track_list:
        meta = copy.deepcopy(track.meta)
        meta.pop("EEPs")
        meta_data_rows.append(tuple(meta.values()))

    if not metatable:
        return track_list
    else:
        print("@mist: making meta table ...")
        the_metatable = Table(rows=meta_data_rows, names=names)
        return track_list, the_metatable


def read_track(fp, colnames=None):
    """ read MIST eep tracks """
    # read lines
    f = open(fp, "r+")
    s = f.readlines()

    # get info
    MIST_version = re.split(r"\s+", s[0].strip())[-1]
    MESA_revision = re.split(r"\s+", s[1].strip())[-1]

    Yinit, Zinit, FeH, aFe, vvcrit = re.split(r"\s+", s[4].strip())[1:]
    Yinit = np.float(Yinit)
    Zinit = np.float(Zinit)
    FeH = np.float(FeH)
    aFe = np.float(aFe)
    vvcrit = np.float(vvcrit)

    initial_mass, N_pts, N_EEP, N_col, phase, type_ = \
        re.split(r"\s+", s[7].strip())[1:]
    initial_mass = np.float(initial_mass)
    N_pts = np.int(N_pts)
    N_EEP = np.int(N_EEP)
    N_col = np.int(N_col)

    # get eep info
    EEPs = tuple([np.int(_) for _ in re.split(r"\s+", s[8].strip())[2:]])
    # eep = np.arange(EEPs[0], EEPs[-1] + 1) sometimes inconsistent with data

    # add eep column
    # _eep
    t = Table.read(s[11:], format="ascii.commented_header")
    eep = np.arange(EEPs[0], EEPs[0] + len(t))
    eep_ok = eep[-1] == EEPs[-1] + 1
    t.add_column(Column(eep, "_eep"))
    # _lgmass
    t.add_column(Column(np.ones(len(t), )*np.log10(initial_mass), "_lgmass"))
    # _lgage
    t.add_column(Column(np.log10(t["star_age"].data), "_lgage"))
    # _feh
    t.add_column(Column(np.ones(len(t), ) * FeH, "_feh_ini"))
    t.add_column(Column(t["log_surf_z"]-np.log10(Zsun), "_feh"))

    # add meta info
    meta = OrderedDict(
        MIST_version=MIST_version,
        MESA_revision=MESA_revision,
        Yinit=Yinit,
        Zinit=Zinit,
        FeH=FeH,
        aFe=aFe,
        vvcrit=vvcrit,
        initial_mass=initial_mass,
        N_pts=N_pts,
        N_EEP=N_EEP,
        N_col=N_col,
        phase=phase,
        type_=type_,
        EEPs=EEPs,
        EEP0=EEPs[0],
        EEP1=EEPs[-1],
        EEP1ACT=EEPs[0] + len(t),
        EEPOK=eep_ok,
        INTERP=("_INTERP" in fp)
    )
    t.meta = meta

    if colnames is None:
        return t
    else:
        for colname in colnames:
            try:
                assert colname in t.colnames
            except AssertionError as ae:
                raise(ae("{} not in track.colnames!!!".format(colname)))
        return t[colnames]
