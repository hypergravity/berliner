import glob
import re
from collections import OrderedDict
from tempfile import NamedTemporaryFile

import numpy as np
from astropy.table import Table, Column
from joblib import Parallel, delayed


def read_isochrones_ptn(ptn="./*.cmd", temp_dir="/tmp/", version="1.2",
                        basic=False, n_jobs=-1, verbose=10):
    """ read multiple MIST cmd files """
    fps = np.sort(glob.glob(ptn))
    if not basic:
        isoc_lists = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(read_isochrones)(fp, temp_dir=temp_dir, version=version)
            for fp in fps)
    else:
        isoc_lists = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(read_isochrones_basic)(fp, temp_dir=temp_dir, version=version)
            for fp in fps)
    return np.hstack(isoc_lists)


def read_isochrones(filepath, temp_dir="/tmp/", version="1.2"):
    """ read MIST isochrones from a directory """
    # 1. read all lines
    with open(filepath) as f:
        s = f.readlines()

    # 2. meta info
    MIST_version = re.split(r"\s+", s[0].strip())[-1]
    MESA_revision = re.split(r"\s+", s[1].strip())[-1]
    photo_sys = re.split(r"=", s[2].strip())[-1].strip()

    print(re.split(r"\s+", s[5].strip())[1:])
    Yinit, Zinit, FeH, aFe, vvcrit = re.split(r"\s+", s[5].strip())[1:]
    Yinit = float(Yinit)
    Zinit = float(Zinit)
    FeH = float(FeH)
    aFe = float(aFe)
    vvcrit = float(vvcrit)
    Niso = int(re.split(r"\s+", s[7].strip())[-1])
    Av = float(re.split(r"\s+", s[8].strip())[-1])

    meta = OrderedDict(MIST_version=MIST_version,
                       MESA_revision=MESA_revision,
                       photo_sys=photo_sys,
                       Yinit=Yinit,
                       Zinit=Zinit,
                       MH=FeH,
                       aFe=aFe,
                       vvcrit=vvcrit,
                       Av=Av)

    # 3. find all isochrones
    istart = []
    for i, _ in enumerate(s):
        if "# number of EEPs, cols =" in _:
            istart.append(i)
    # check number of isochrones
    assert len(istart) == Niso

    # determine the start and stop lines of each isochrones
    istart = np.array(istart)
    istop = np.roll(istart, -1)
    istop[-1] = -1
    istart += 2

    # 4. read all isochrones
    isolist = []
    import os
    tfn = NamedTemporaryFile(dir=temp_dir, delete=False).name
    for iiso in range(Niso):
        with open(tfn, "w+") as f:
            f.writelines(s[istart[iiso]:istop[iiso]])
        iso = Table.read(tfn, format="ascii.fast_commented_header")
        iso.meta = meta
        iso.meta["lgage"] = np.unique(iso["log10_isochrone_age_yr"].data)[0]
        isolist.append(iso)
        os.remove(tfn)

    # 5. make basic columns
    if version == "1.1" or version == "1.2":
        for i in range(len(isolist)):
            _lgmass = Column(np.log10(isolist[i]["initial_mass"]), "_lgmass")
            _mhini = Column(isolist[i]["[Fe/H]_init"], "_mhini")
            _lgage = Column(isolist[i]["log10_isochrone_age_yr"], "_lgage")
            _eep = Column(isolist[i]["EEP"], "_eep")
            isolist[i].add_columns((_lgmass, _mhini, _lgage, _eep))
    elif version == "1.0":
        for i in range(len(isolist)):
            _lgmass = Column(np.log10(isolist[i]["initial_mass"]), "_lgmass")
            _mhini = Column(np.log10(isolist[i]["Z_surf"]), "_mhini")
            _lgage = Column(isolist[i]["log10_isochrone_age_yr"], "_lgage")
            _eep = Column(isolist[i]["EEP"], "_eep")
            isolist[i].add_columns((_lgmass, _mhini, _lgage, _eep))

    return isolist


def read_isochrones_basic(filepath, temp_dir="/tmp/", version="1.2"):
    """ read MIST isochrones from a directory """
    # 1. read all lines
    with open(filepath) as f:
        s = f.readlines()

    # 2. meta info
    MIST_version = re.split(r"\s+", s[0].strip())[-1]
    MESA_revision = re.split(r"\s+", s[1].strip())[-1]
    # photo_sys = re.split(r"=", s[2].strip())[-1].strip()

    # print(re.split(r"\s+", s[5].strip())[1:])
    Yinit, Zinit, FeH, aFe, vvcrit = re.split(r"\s+", s[4].strip())[1:]
    Yinit = float(Yinit)
    Zinit = float(Zinit)
    FeH = float(FeH)
    aFe = float(aFe)
    vvcrit = float(vvcrit)
    Niso = int(re.split(r"\s+", s[6].strip())[-1])
    # Av = float(re.split(r"\s+", s[7].strip())[-1])

    meta = OrderedDict(MIST_version=MIST_version,
                       MESA_revision=MESA_revision,
                       # photo_sys=photo_sys,
                       Yinit=Yinit,
                       Zinit=Zinit,
                       MH=FeH,
                       aFe=aFe,
                       vvcrit=vvcrit,
                       # Av=Av,
                       )

    # 3. find all isochrones
    istart = []
    for i, _ in enumerate(s):
        if "# number of EEPs, cols =" in _:
            istart.append(i)
    # check number of isochrones
    assert len(istart) == Niso

    # determine the start and stop lines of each isochrones
    istart = np.array(istart)
    istop = np.roll(istart, -1)
    istop[-1] = -1
    istart += 2

    # 4. read all isochrones
    isolist = []
    import os
    tfn = NamedTemporaryFile(dir=temp_dir, delete=False).name
    for iiso in range(Niso):
        with open(tfn, "w+") as f:
            f.writelines(s[istart[iiso]:istop[iiso]])
        iso = Table.read(tfn, format="ascii.fast_commented_header")
        iso.meta = meta
        iso.meta["lgage"] = np.unique(iso["log10_isochrone_age_yr"].data)[0]
        isolist.append(iso)
        os.remove(tfn)

    # 5. make basic columns
    if version == "1.1" or version == "1.2":
        for i in range(len(isolist)):
            _lgmass = Column(np.log10(isolist[i]["initial_mass"]), "_lgmass")
            _mhini = Column(np.ones((len(isolist[i], )), float) * FeH, "_mhini")
            _lgage = Column(isolist[i]["log10_isochrone_age_yr"], "_lgage")
            _eep = Column(isolist[i]["EEP"], "_eep")
            isolist[i].add_columns((_lgmass, _mhini, _lgage, _eep))
    elif version == "1.0":
        for i in range(len(isolist)):
            _lgmass = Column(np.log10(isolist[i]["initial_mass"]), "_lgmass")
            _mhini = Column(np.log10(isolist[i]["Z_surf"]), "_mhini")
            _lgage = Column(isolist[i]["log10_isochrone_age_yr"], "_lgage")
            _eep = Column(isolist[i]["EEP"], "_eep")
            isolist[i].add_columns((_lgmass, _mhini, _lgage, _eep))

    return isolist
