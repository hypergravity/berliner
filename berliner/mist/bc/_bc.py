import glob
import os

import joblib
import numpy as np
from astropy.table import Table, vstack
from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator

# ######################### #
# read BC from Conroy et al #
# ######################### #
mapdict = {
    'CFHTugriz': ['CFHT_u', 'CFHT_g', 'CFHT_r', 'CFHT_i_new', 'CFHT_i_old',
                  'CFHT_z'],
    'DECam': ['DECam_u', 'DECam_g', 'DECam_r', 'DECam_i', 'DECam_z',
              'DECam_Y'], 'GALEX': ['GALEX_FUV', 'GALEX_NUV'],
    'HST_ACSHR': ['ACS_HRC_F220W', 'ACS_HRC_F250W', 'ACS_HRC_F330W',
                  'ACS_HRC_F344N', 'ACS_HRC_F435W', 'ACS_HRC_F475W',
                  'ACS_HRC_F502N', 'ACS_HRC_F550M', 'ACS_HRC_F555W',
                  'ACS_HRC_F606W', 'ACS_HRC_F625W', 'ACS_HRC_F658N',
                  'ACS_HRC_F660N', 'ACS_HRC_F775W', 'ACS_HRC_F814W',
                  'ACS_HRC_F850LP', 'ACS_HRC_F892N'],
    'HST_ACSWF': ['ACS_WFC_F435W', 'ACS_WFC_F475W', 'ACS_WFC_F502N',
                  'ACS_WFC_F550M', 'ACS_WFC_F555W', 'ACS_WFC_F606W',
                  'ACS_WFC_F625W', 'ACS_WFC_F658N', 'ACS_WFC_F660N',
                  'ACS_WFC_F775W', 'ACS_WFC_F814W', 'ACS_WFC_F850LP',
                  'ACS_WFC_F892N'],
    'HST_WFC3': ['WFC3_UVIS_F200LP', 'WFC3_UVIS_F218W', 'WFC3_UVIS_F225W',
                 'WFC3_UVIS_F275W', 'WFC3_UVIS_F280N', 'WFC3_UVIS_F300X',
                 'WFC3_UVIS_F336W', 'WFC3_UVIS_F343N', 'WFC3_UVIS_F350LP',
                 'WFC3_UVIS_F373N', 'WFC3_UVIS_F390M', 'WFC3_UVIS_F390W',
                 'WFC3_UVIS_F395N', 'WFC3_UVIS_F410M', 'WFC3_UVIS_F438W',
                 'WFC3_UVIS_F467M', 'WFC3_UVIS_F469N', 'WFC3_UVIS_F475W',
                 'WFC3_UVIS_F475X', 'WFC3_UVIS_F487N', 'WFC3_UVIS_F502N',
                 'WFC3_UVIS_F547M', 'WFC3_UVIS_F555W', 'WFC3_UVIS_F600LP',
                 'WFC3_UVIS_F606W', 'WFC3_UVIS_F621M', 'WFC3_UVIS_F625W',
                 'WFC3_UVIS_F631N', 'WFC3_UVIS_F645N', 'WFC3_UVIS_F656N',
                 'WFC3_UVIS_F657N', 'WFC3_UVIS_F658N', 'WFC3_UVIS_F665N',
                 'WFC3_UVIS_F673N', 'WFC3_UVIS_F680N', 'WFC3_UVIS_F689M',
                 'WFC3_UVIS_F763M', 'WFC3_UVIS_F775W', 'WFC3_UVIS_F814W',
                 'WFC3_UVIS_F845M', 'WFC3_UVIS_F850LP', 'WFC3_UVIS_F953N',
                 'WFC3_IR_F098M', 'WFC3_IR_F105W', 'WFC3_IR_F110W',
                 'WFC3_IR_F125W', 'WFC3_IR_F126N', 'WFC3_IR_F127M',
                 'WFC3_IR_F128N', 'WFC3_IR_F130N', 'WFC3_IR_F132N',
                 'WFC3_IR_F139M', 'WFC3_IR_F140W', 'WFC3_IR_F153M',
                 'WFC3_IR_F160W', 'WFC3_IR_F164N', 'WFC3_IR_F167N'],
    'HST_WFPC2': ['WFPC2_F218W', 'WFPC2_F255W', 'WFPC2_F300W', 'WFPC2_F336W',
                  'WFPC2_F439W', 'WFPC2_F450W', 'WFPC2_F555W', 'WFPC2_F606W',
                  'WFPC2_F622W', 'WFPC2_F675W', 'WFPC2_F791W', 'WFPC2_F814W',
                  'WFPC2_F850LP'],
    'JWST': ['F070W', 'F090W', 'F115W', 'F140M', 'F150W2', 'F150W', 'F162M',
             'F164N', 'F182M', 'F187N', 'F200W', 'F210M', 'F212N', 'F250M',
             'F277W', 'F300M', 'F322W2', 'F323N', 'F335M', 'F356W', 'F360M',
             'F405N', 'F410M', 'F430M', 'F444W', 'F460M', 'F466N', 'F470N',
             'F480M'],
    'LSST': ['LSST_u', 'LSST_g', 'LSST_r', 'LSST_i', 'LSST_z', 'LSST_y'],
    'PanSTARRS': ['PS_g', 'PS_r', 'PS_i', 'PS_z', 'PS_y', 'PS_w', 'PS_open'],
    'SDSSugriz': ['SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z'],
    'SPITZER': ['IRAC_3.6', 'IRAC_4.5', 'IRAC_5.8', 'IRAC_8.0'],
    'SkyMapper': ['SkyMapper_u', 'SkyMapper_v',
                  'SkyMapper_g', 'SkyMapper_r',
                  'SkyMapper_i', 'SkyMapper_z'],
    'Swift': ['Swift_UVW2', 'Swift_UVM2', 'Swift_UVW1', 'Swift_U', 'Swift_B',
              'Swift_V'],
    'UBVRIplus': ['Bessell_U', 'Bessell_B', 'Bessell_V', 'Bessell_R',
                  'Bessell_I', '2MASS_J', '2MASS_H', '2MASS_Ks',
                  'Kepler_Kp', 'Kepler_D51', 'Hipparcos_Hp', 'Tycho_B',
                  'Tycho_V', 'Gaia_G', 'Gaia_BP', 'Gaia_RP'],
    'UKIDSS': ['UKIDSS_Z', 'UKIDSS_Y', 'UKIDSS_J', 'UKIDSS_H', 'UKIDSS_K'],
    'WISE': ['WISE_W1', 'WISE_W2', 'WISE_W3', 'WISE_W4'],
    'WashDDOuvby': ['Washington_C', 'Washington_M', 'Washington_T1',
                    'Washington_T2', 'DDO51_vac', 'DDO51_f31', 'Stromgren_u',
                    'Stromgren_v', 'Stromgren_b', 'Stromgren_y']}


def combine_bctables(bctables):
    result = bctables[0]
    for _bctable in bctables[1:]:
        for colname in _bctable.colnames:
            if not colname in result.colnames:
                result.add_column(_bctable[colname])

    return result


def read_bc(bc_fp):
    """ read a single BC file """
    with open(bc_fp, "r+") as f:
        s = f.readlines()
    data = Table.read(s[5:], format="ascii.commented_header")

    data.meta["INSTRUMENT"] = (s[0]).strip()[2:]
    data.meta["FILTERS"] = data.colnames[5:]
    return data


def read_bc_dir(bc_dir):
    """ read all BC files in a directory """
    data = []
    for fp in glob.glob(bc_dir + "/*"):
        if os.path.isfile(fp):
            data.append(read_bc(fp))

    # stack tables
    result = vstack(data)
    result.meta = data[0].meta
    return result


def read_bc_dir_one(bc_dir):
    """ read only the first one file """
    fp = glob.glob(bc_dir + "/*")[0]
    return read_bc(fp)


def glob_subdirs(pattern):
    sub_dirs_ = glob.glob(pattern)
    sub_dirs = [_ for _ in sub_dirs_ if os.path.isdir(_)]
    return sub_dirs


def make_mapdict(bc_root="/home/cham/projects/gaia/data/mist/BC"):
    """ make a mapping dict for multiple direcotries(filters) of BC files """
    bc_dirs = glob_subdirs(bc_root + "/*")
    mapdict = dict()
    for bc_dir in bc_dirs:
        print(bc_dir)
        x = read_bc_dir_one(bc_dir)
        mapdict[os.path.basename(bc_dir)] = x.meta["FILTERS"]
    return mapdict


def make_imapdict(mapdict):
    """ make inverse mapping dict """
    result = dict()
    for k, v in mapdict.items():
        for _ in v:
            result[_] = k

    return result


def init_mapdict(bc_root="/home/cham/projects/gaia/data/mist/BC"):
    """ make mapping & inverse mapping dicts for a given root directory """
    mapdict = make_mapdict(bc_root)
    imapdict = make_imapdict(mapdict)
    return mapdict, imapdict


def make_bc(bc_root="/home/cham/projects/gaia/data/mist/BC", filter="Gaia_G",
            kind="linear", logteff=False):
    """ make BC interpolator
    Parameters:
    -----------
    bc_root:
        root dir of BC files

    filter:
        filter name
    kind:
        ["linear", "nearest"]

    Returns:
    --------
    linear/cubic interpolator for BC data,
    basic parameters are: ['Teff', 'logg', '[Fe/H]', 'Av'] assuming Rv=3.1

    """
    mapdict, imapdict = init_mapdict(bc_root)
    if filter not in imapdict.keys():
        raise ValueError("Filter [{}] is not available!".format(filter))
    bc_dir = bc_root + "/" + imapdict[filter]
    data = read_bc_dir(bc_dir)

    if kind == "linear":
        # points = np.array(data['Teff', 'logg', '[Fe/H]', 'Av'].to_pandas())
        # values = np.array(data[filter])
        # lndi = LinearNDInterpolator(points, values, fill_value=np.nan,
        #                             rescale=True)
        # rgi = RegularGridInterpolator(points, values)
        u_teff = np.unique(data["Teff"])
        u_logg = np.unique(data["logg"])
        u_feh = np.unique(data["[Fe/H]"])
        u_av = np.unique(data["Av"])
        values = np.zeros((len(u_teff), len(u_logg), len(u_feh), len(u_av)))
        for i in range(len(data)):
            ind_teff = u_teff == data["Teff"][i]
            ind_logg = u_logg == data["logg"][i]
            ind_feh = u_feh == data["[Fe/H]"][i]
            ind_av = u_av == data["Av"][i]
            values[ind_teff, ind_logg, ind_feh, ind_av] = data[filter][i]

        if logteff:
            rgi = RegularGridInterpolator(
                (u_teff, u_logg, u_feh, u_av), values, bounds_error=False,
                fill_value=np.nan)
        else:
            rgi = RegularGridInterpolator(
                (np.log10(u_teff), u_logg, u_feh, u_av), values,
                bounds_error=False, fill_value=np.nan)
        return rgi
    else:
        return None


def make_bc_dir(bc_dir="/home/cham/projects/gaia/data/mist/BC/CFHTugriz",
                kind="linear", logteff=False, save_dir=None):
    """ make BC interpolator
    Parameters:
    -----------
    bc_root:
        root dir of BC files

    filter:
        filter name
    kind:
        ["linear", "nearest"]

    Returns:
    --------
    linear/cubic interpolator for BC data,
    basic parameters are: ['Teff', 'logg', '[Fe/H]', 'Av'] assuming Rv=3.1

    """
    # default saving dir
    if save_dir is None:
        save_dir = os.path.dirname(bc_dir)

    # read BC data
    # print("@BC: reading BC data ...")
    data = read_bc_dir(bc_dir)

    # make rgi for all filters in this BC
    for filter in data.meta["FILTERS"]:
        save_fp = save_dir + "/" + filter + ".dump"
        if kind == "linear":
            # points = np.array(data['Teff', 'logg', '[Fe/H]', 'Av'].to_pandas())
            # values = np.array(data[filter])
            # lndi = LinearNDInterpolator(points, values, fill_value=np.nan,
            #                             rescale=True)
            # rgi = RegularGridInterpolator(points, values)
            u_teff = np.unique(data["Teff"])
            u_logg = np.unique(data["logg"])
            u_feh = np.unique(data["[Fe/H]"])
            u_av = np.unique(data["Av"])
            values = np.zeros(
                (len(u_teff), len(u_logg), len(u_feh), len(u_av)))
            for i in range(len(data)):
                ind_teff = u_teff == data["Teff"][i]
                ind_logg = u_logg == data["logg"][i]
                ind_feh = u_feh == data["[Fe/H]"][i]
                ind_av = u_av == data["Av"][i]
                values[ind_teff, ind_logg, ind_feh, ind_av] = data[filter][i]

            if logteff:
                rgi = RegularGridInterpolator(
                    (np.log10(u_teff), u_logg, u_feh, u_av), values,
                    bounds_error=False, fill_value=np.nan)
            else:
                rgi = RegularGridInterpolator(
                    (u_teff, u_logg, u_feh, u_av), values,
                    bounds_error=False, fill_value=np.nan)
            print("@BC: saving [{}] --> *{}* ...".format(filter, save_fp))
            joblib.dump(rgi, save_fp)
        else:
            return None


def make_bc_all(bc_root="/home/cham/projects/gaia/data/mist/BC",
                logteff=False, save_dir=None, n_jobs=-1, verbose=20):
    """ make BC rgi in parallel """
    from berliner.mist.bc._bc import make_bc_dir, init_mapdict
    mapdict, imapdict = init_mapdict(bc_root)
    bc_dirs = glob_subdirs(bc_root + "/*")
    print("@BC: BC directories:")
    print(bc_dirs)
    print("")

    Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(make_bc_dir)(
        bc_dir=bc_dir, logteff=logteff, save_dir=save_dir) \
        for bc_dir in bc_dirs)
    return


class BCI:
    bc_root = ""
    rgi = []
    logteff = False
    init_status = False
    afilters = []
    adumpfps = []
    nfilters = 0

    loaded_filters = []

    mapdict = dict()

    def __init__(self,
                 bc_root="/media/cham/Seagate Expansion Drive/mist11/BC/rgi_linteff",
                 logteff=False):
        self.adumpfps = glob.glob(bc_root + "/*.dump")
        self.afilters = [os.path.splitext(os.path.basename(bc_dump_fp))[0] for
                         bc_dump_fp in self.adumpfps]
        self.nfilters = len(self.afilters)
        for i in range(self.nfilters):
            self.mapdict[self.afilters[i]] = self.adumpfps[i]

        self.logteff = logteff
        return

    def load_bc(self, filters=["Gaia_G"], bc_root=None):
        # assert filter exists
        assert len(filters) > 0
        self.rgi = []
        for filter in filters:
            try:
                assert filter in self.afilters
                self.rgi.append(joblib.load(self.mapdict[filter]))
            except AssertionError as ae:
                raise AssertionError(
                    "Filter [{}] is not available!".format(filter))

        self.loaded_filters = np.copy(filters)
        return

    def __call__(self, points, *args, **kwargs):
        """ points = a set of (Teff, logg, [Fe/H], Av) """
        if points.ndim == 1:
            points = points.reshape(1, -1)

        if self.logteff:
            points[:, 0] = 10. ** points[:, 0]

        return np.array([rgi(points) for rgi in self.rgi]).T

    def interp_mag(self, Mbol, teff, logg, feh, av=0.):
        return Mbol - self(np.array([teff, logg, feh, av]))[0]
