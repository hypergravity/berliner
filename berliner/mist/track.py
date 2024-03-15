import glob
import re
from collections import OrderedDict
import copy
import warnings

import numpy as np
import numpy.typing as npt
from astropy.table import Table, Column
from joblib import Parallel, delayed

ZSUN = 0.0142857  # ref:


def load_tracks(
    pattern: str = "./*.isochrone.eep",
    n_jobs: int = -1,
    verbose: int = 5,
) -> tuple[list[Table], npt.NDArray]:
    """read multiple MIST isochrone files"""
    fps = np.sort(glob.glob(pattern))
    print(f"@mist: reading {len(fps)} tracks ...")
    track_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(load_track)(fp) for fp in fps
    )
    params = np.array([track.meta["params"] for track in track_list])
    # meta0 = copy.deepcopy(track_list[0].meta)
    # meta0.pop("EEPs")
    # names = tuple(meta0.keys())
    #
    # meta_data_rows = []
    # for isochrone in track_list:
    #     meta = copy.deepcopy(isochrone.meta)
    #     meta.pop("EEPs")
    #     meta_data_rows.append(tuple(meta.values()))
    #
    # if not metatable:
    #     return track_list
    # else:
    #     print("@mist: making meta table ...")
    #     the_metatable = Table(rows=meta_data_rows, names=names)
    #     return track_list, the_metatable
    return track_list, params


def load_track(fp: str) -> Table:
    """read MIST eep tracks"""
    # read lines
    # print(f"reading isochrone ... {fp}")
    with open(fp, "r") as f:
        s = f.readlines()

    # get info
    MIST_version = re.split(r"\s+", s[0].strip())[-1]
    MESA_revision = re.split(r"\s+", s[1].strip())[-1]

    Yinit, Zinit, FeH, aFe, vvcrit = re.split(r"\s+", s[4].strip())[1:]
    Yinit = float(Yinit)
    Zinit = float(Zinit)
    FeH = float(FeH)
    aFe = float(aFe)
    vvcrit = float(vvcrit)

    initial_mass, N_pts, N_EEP, N_col, phase, type_ = re.split(r"\s+", s[7].strip())[1:]
    # print(initial_mass, N_pts, N_EEP, N_col, phase, type_)
    initial_mass = float(initial_mass)
    N_pts = int(N_pts)
    N_EEP = int(N_EEP)
    N_col = int(N_col)

    # get eep info
    EEPs = tuple([int(_) for _ in re.split(r"\s+", s[8].strip())[2:]])
    # eep = np.arange(EEPs[0], EEPs[-1] + 1) sometimes inconsistent with data

    # add eep column
    # eep
    t = Table.read(s[11:], format="ascii.commented_header")
    eep = np.arange(EEPs[0], EEPs[0] + len(t))
    eep_consistency = eep[-1] == EEPs[-1]

    if not eep_consistency:
        warnings.warn(f"Inconsistent EEPs for {fp}", category=UserWarning)

    t.add_column(Column(eep, "eep"))
    # log_mass
    t.add_column(Column(np.ones(len(t)) * np.log10(initial_mass), "log_mass"))
    # log_age
    t.add_column(Column(np.log10(t["star_age"].data), "log_age"))
    # feh_ini feh
    t.add_column(Column(np.ones(len(t)) * FeH, "feh_ini"))
    t.add_column(Column(t["log_surf_z"] - np.log10(ZSUN), "feh"))

    # add meta info
    meta = OrderedDict(
        params=[initial_mass, FeH],
        fp=fp,
        eep_consistency=eep_consistency,
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
        type=type_,
        EEPs=EEPs,
        EEP0=EEPs[0],
        EEP1=EEPs[-1],
        EEP1ACT=EEPs[0] + len(t) - 1,
        INTERP=("_INTERP" in fp),
    )
    t.meta = meta
    return t
