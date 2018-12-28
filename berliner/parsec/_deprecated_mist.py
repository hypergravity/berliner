import numpy as np
import re
from tempfile import NamedTemporaryFile
from astropy.table import Table, Column
from collections import OrderedDict
# from .isochrone import Isochrone
import glob


def mist_eep_track_filenames(fmt="./*/*.eep*"):
    """ return mist eep track filenames """
    # find all track files
    fps = glob.glob(fmt)

    fps_bad = []
    # eliminate bad tracks
    for fp in fps:
        if "_INTERP" in fp:
            fps_bad.append(fp.replace("_INTERP", ""))
    for fp_bad in fps_bad:
        try:
            fps.remove(fp_bad)
            print("{} is removed ...".format(fp_bad))
        except ValueError as ve:
            print("{} not found ...".format(fp_bad))

    return fps


def sign(num):
    if num >= 0:
        return "p"
    else:
        return "m"


def make_new_name(track, dirname=None):
    """ make a new name for a track """
    mist_version = track.meta["MIST_version"]
    feh = "{:0.2f}".format(np.abs(track.meta["FeH"]))
    sign_feh = sign(track.meta["FeH"])
    afe = "{:.1f}".format(track.meta["aFe"])
    sign_afe = sign(track.meta["aFe"])
    vvcrit = "{:.1}".format(track.meta["vvcrit"])

    initial_mass = "{:.0f}".format(track.meta["initial_mass"] * 100).zfill(5)

    new_name = "MIST_v{}_feh_{}{}_afe_{}{}_vvcrit{}_EEPS.{}M.track.eep.fits".format(
        mist_version, sign_feh, feh, sign_afe, afe, vvcrit, initial_mass)
    # interp = "{}".format(track.meta["INTERP"])
    if dirname is None:
        return new_name
    else:
        if dirname[-1] != "/":
            dirname += "/"
        return dirname + "/" + new_name


def read_eep_track(fp, colnames=None):
    """ read MIST eep tracks """
    # read lines
    f = open(fp, "r+")
    s = f.readlines()

    # get info
    MIST_version = re.split(r"\s+", s[0].strip())[-1]
    MESA_revision = re.split(r"\s*", s[1].strip())[-1]

    Yinit, Zinit, FeH, aFe, vvcrit = re.split(r"\s*", s[4].strip())[1:]
    Yinit = np.float(Yinit)
    Zinit = np.float(Zinit)
    FeH = np.float(FeH)
    aFe = np.float(aFe)
    vvcrit = np.float(vvcrit)

    initial_mass, N_pts, N_EEP, N_col, phase, type_ = \
        re.split(r"\s*", s[7].strip())[1:]
    initial_mass = np.float(initial_mass)
    N_pts = np.int(N_pts)
    N_EEP = np.int(N_EEP)
    N_col = np.int(N_col)

    # get eep info
    EEPs = [np.int(_) for _ in re.split(r"\s+", s[8].strip())[2:]]
    eep = np.arange(EEPs[0], EEPs[-1] + 1)

    # add eep column
    # _eep
    t = Table.read(s[11:], format="ascii.commented_header")
    t.add_column(Column(eep, "_eep"))
    # _lgmass
    t.add_column(Column(np.ones(len(t), )*np.log10(initial_mass), "_lgmass"))
    # _lgage
    t.add_column(Column(np.log10(t["star_age"].data), "_lgage"))
    # _feh
    t.add_column(Column(np.ones(len(t), ) * FeH, "_feh"))

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
        return t


def read_mist(filepath, temp_dir="/home/cham/.isochrones/mist11/MIST_v1.1_SDSS/",
              version="1.1"):
    # 1. read all lines
    with open(filepath) as f:
        s = f.readlines()

    # 2. meta info
    MIST_version = re.split(r"\s+", s[0].strip())[-1]
    MESA_revision = re.split(r"\s*", s[1].strip())[-1]
    photo_sys = re.split(r"=", s[2].strip())[-1].strip()

    Yinit, Zinit, FeH, aFe, vvcrit = re.split(r"\s*", s[5].strip())[1:]
    Yinit = np.float(Yinit)
    Zinit = np.float(Zinit)
    FeH = np.float(FeH)
    aFe = np.float(aFe)
    vvcrit = np.float(vvcrit)
    Niso = np.int(re.split(r"\s+", s[7].strip())[-1])
    Av = np.float(re.split(r"\s+", s[8].strip())[-1])

    meta = OrderedDict(MIST_version=MIST_version,
                       MESA_revision=MESA_revision,
                       photo_sys=photo_sys,
                       Yinit=Yinit,
                       Zinit=Zinit,
                       FeH=FeH,
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
    if version == "1.1":
        for i in range(len(isolist)):
            _lgmass = Column(np.log10(isolist[i]["initial_mass"]), "_lgmass")
            _feh = Column(isolist[i]["[Fe/H]_init"], "_feh")
            _lgage = Column(isolist[i]["log10_isochrone_age_yr"], "_lgage")
            _eep = Column(isolist[i]["EEP"], "_eep")
            isolist[i].add_columns((_lgmass, _feh, _lgage, _eep))
    elif version == "1.0":
        for i in range(len(isolist)):
            _lgmass = Column(np.log10(isolist[i]["initial_mass"]), "_lgmass")
            _feh = Column(np.log10(isolist[i]["Z_surf"]), "_feh")
            _lgage = Column(isolist[i]["log10_isochrone_age_yr"], "_lgage")
            _eep = Column(isolist[i]["EEP"], "_eep")
            isolist[i].add_columns((_lgmass, _feh, _lgage, _eep))

    return isolist