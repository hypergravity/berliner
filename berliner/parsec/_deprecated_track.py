import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
from astropy import table
from astropy.table import Table, Column
import warnings




def get_track_meta(track, key="FeH"):
    """ get meta info from a track """
    assert key in track.meta.keys()
    return track.meta[key]


def find_rank_1d(arr, val, sort=False):
    """ return ind of the two elements in *arr* that bracket *val* """
    if sort:
        arr = np.sort(arr)
    sub = np.where((arr[:-1] < val) & (arr[1:] >= val))[0]
    assert len(sub) > 0

    return np.hstack((sub, sub + 1))


def get_track_item_given_eeps(track, eeps):
    """ return track items given a set of eeps """
    ind = np.zeros((len(track, )), dtype=bool)
    for eep in eeps:
        ind |= track["_eep"] == eep
    return track[ind]


def calc_weight(arr, val, norm=1.):
    """ calculate normalized weight """
    weight = np.abs(arr - val)
    weight *= norm / np.sum(weight)
    return np.array(weight)


def table_linear_combination(t, weight):
    """ given weight, return the linear combination of each row for each column
    """
    assert len(t) == len(weight)
    new_cols = []
    colnames = t.colnames
    ncols = len(colnames)
    for i in range(ncols):
        if t.dtype[i] in (np.int, np.float):
            colname = colnames[i]
            new_cols.append(
                Column(np.array([np.sum(t[colname].data * weight)]), colname))
    return Table(new_cols)


class StarObject():
    def __init__(self, t):
        assert len(t) == 1
        colnames = t.colnames
        ncols = len(colnames)
        for i in range(ncols):
            self.__setattr__(colnames[i], t[colnames[i]].data[0])


class TrackSet:
    """ a set of tracks """

    data = []
    eep_bounds = (1, 808)

    default_coord = ["_lgmass", "_feh", "_lgage", "_eep"]

    bci = None

    def __init__(self, tracks,
                 metadict=dict(minit="initial_mass",
                               feh="FEH",
                               eep="EEPS",
                               mbol="Mbol")):
        """ initialization of track set object """

        self.metadict = metadict
        self.data = np.array(tracks)

        self.grid_minit = np.array(
            [get_track_meta(track, metadict["minit"]) for track in tracks])
        self.grid_feh = np.array(
            [get_track_meta(track, metadict["feh"]) for track in tracks])
        #self.grid_EEP = [get_track_meta(track, metadict["eep"]) for track in tracks]

        # every track starts from EEP=1
        self.grid_EEP0 = np.array([np.min(_["_eep"]) for _ in self.data])
        self.grid_EEP1 = np.array([np.max(_["_eep"]) for _ in self.data])

        self.u_minit = np.unique(self.grid_minit)
        self.u_feh = np.unique(self.grid_feh)

        self.min_minit = np.min(self.u_minit)
        self.max_minit = np.max(self.u_minit)
        self.min_feh = np.min(self.u_feh)
        self.max_feh = np.max(self.u_feh)
        self.min_eep = np.min(self.grid_EEP0)
        self.max_eep = np.max(self.grid_EEP1)

    def get_track4(self, mass_feh=(1.01, 0.01)):
        """ return the 4 neighboring stellar tracks """
        test_minit, test_feh = np.array(mass_feh, dtype=np.float)

        # assert Minit [Fe/H] in range
        try:
            assert self.min_minit < test_minit <= self.max_minit
            assert self.min_feh < test_feh <= self.max_feh
        except AssertionError as ae:
            return None

        # 1. locate 4 tracks
        ind_minit = find_rank_1d(self.u_minit, test_minit)
        ind_feh = find_rank_1d(self.u_feh, test_feh)
        val_minit = self.u_minit[ind_minit]
        val_feh = self.u_feh[ind_feh]

        ind_track = np.where(np.logical_and(
            (self.grid_minit == val_minit[0]) | (
            self.grid_minit == val_minit[1]),
            (self.grid_feh == val_feh[0]) | (self.grid_feh == val_feh[1])))[0]
        track4 = self.data[ind_track]

        return track4

    def get_track4_unstructured(self, mass_feh=(1.01, 0.01)):
        """ return the 4 neighboring stellar tracks given unstructured grid """
        test_minit, test_feh = np.array(mass_feh, dtype=np.float)

        d_minit_feh = (np.log10(self.grid_minit)-np.log10(test_minit))**2. + \
                      (self.grid_feh - test_feh) ** 2.
        mask00 = (self.grid_minit < test_minit) & (self.grid_feh < test_feh)
        mask01 = (self.grid_minit < test_minit) & (self.grid_feh >= test_feh)
        mask10 = (self.grid_minit >= test_minit) & (self.grid_feh < test_feh)
        mask11 = (self.grid_minit >= test_minit) & (self.grid_feh >= test_feh)
        if np.any(np.array([np.sum(mask00), np.sum(mask01),
                            np.sum(mask10), np.sum(mask11)]) == 0):
            return None
        ind00 = np.argmin(np.ma.MaskedArray(d_minit_feh, ~mask00))
        ind01 = np.argmin(np.ma.MaskedArray(d_minit_feh, ~mask01))
        ind10 = np.argmin(np.ma.MaskedArray(d_minit_feh, ~mask10))
        ind11 = np.argmin(np.ma.MaskedArray(d_minit_feh, ~mask11))

        return self.data[[ind00, ind01, ind10, ind11]]

    def interp_mass_feh_eep(self, interp_colname="_lgage",
                            mfe=(1.01, 0.01, 503.2),
                            lndi=True, debug=False, raise_error=False):
        test_minit, test_feh, test_eep = np.array(mfe, dtype=np.float)

        # 1. assert Minit [Fe/H] in range
        try:
            assert self.min_minit < test_minit <= self.max_minit
            assert self.min_feh < test_feh <= self.max_feh
        except AssertionError as ae:
            if not raise_error:
                return np.nan
            else:
                raise ae("The test values are not in bounds!")

        # 2. locate 4 tracks
        # ind_minit = find_rank_1d(self.u_minit, test_minit)
        # ind_feh = find_rank_1d(self.u_feh, test_feh)
        # val_minit = self.u_minit[ind_minit]
        # val_feh = self.u_feh[ind_feh]
        #
        # ind_track = np.where(np.logical_and(
        #     (self.grid_minit == val_minit[0]) | (self.grid_minit == val_minit[1]),
        #     (self.grid_feh == val_feh[0]) | (self.grid_feh == val_feh[1])))[0]
        # track4 = self.data[ind_track]
        track4 = self.get_track4_unstructured((test_minit, test_feh))
        if track4 is None:
            if raise_error:
                raise(ValueError("Bad test values!"))
            else:
                return np.nan
        eep_maxmin = np.max([_["_eep"][0] for _ in track4])
        eep_minmax = np.min([_["_eep"][-1] for _ in track4])

        # 3. assert EEP in range
        try:
            assert eep_maxmin < test_eep <= eep_minmax
        except AssertionError as ae:
            if not raise_error:
                return np.nan
            else:
                raise ae("EEP value is not in bounds!")

        # 4. locate EEP
        eep_arr = np.arange(eep_maxmin, eep_minmax + 1)
        ind_eep = find_rank_1d(eep_arr, test_eep)
        val_eep = eep_arr[ind_eep]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            track_box = table.vstack([
                get_track_item_given_eeps(track, val_eep) for track in track4])

        # 5. interpolate
        if not lndi:
            # points = np.array(track_box["_lgmass", "_feh", "_eep"].to_pandas())
            # values = track_box[interp_colname].data
            # lndi = LinearNDInterpolator(points, values)
            # test_points = np.array((np.log10(test_minit), test_feh, test_eep))
            w_mfe = (1 - calc_weight(track_box["_lgmass"], np.log10(test_minit), 4)) * \
                    (1 - calc_weight(track_box["_feh"], test_feh, 4)) * \
                    (1 - calc_weight(track_box["_eep"], test_eep, 4))
            if debug:
                return w_mfe
            star_result = table_linear_combination(track_box, w_mfe)
            return star_result

        elif type(interp_colname) is not list:
            points = np.array(track_box["_lgmass", "_feh", "_eep"].to_pandas())
            # for linear mass
            # points[:, 0] = np.power(10, points[:, 0])
            values = track_box[interp_colname].data
            lndi = LinearNDInterpolator(points, values)
            test_points = np.array((np.log10(test_minit), test_feh, test_eep))
            # for linear mass
            # test_points = np.array((np.log10(test_minit), test_feh, test_eep))
            return lndi(test_points)[0]
        elif type(interp_colname) is list:
            points = np.array(track_box["_lgmass", "_feh", "_eep"].to_pandas())
            test_points = np.array((np.log10(test_minit), test_feh, test_eep))
            results = []
            for _interp_colname in interp_colname:
                if type(_interp_colname) is int:
                    # directly return the input value if int
                    results.append(test_points[_interp_colname])
                else:
                    values = track_box[_interp_colname].data
                    results.append(
                        LinearNDInterpolator(points, values)(test_points)[0])
            return np.array(results)

    def calc_dlgagedeep(self, track, deep=0.1):
        I = interp1d(track["_eep"], track["_lgage"], kind="linear",
                     bounds_error=False, fill_value=-np.inf)
        dlgagedeep = (I(track["_eep"] + deep) - I(track["_eep"] - deep)) \
                     / deep / 2.
        track.add_column(Column(dlgagedeep, "dlgagedeep"))
        return track

    def calc_dlgagedeep_for_all_tracks(self, deep=0.1):
        for track in self.data:

            I = interp1d(track["_eep"], track["_lgage"], kind="linear",
                         bounds_error=False, fill_value=-np.inf)
            dlgagedeep = (I(track["_eep"] + deep) -
                          I(track["_eep"] - deep)) / deep / 2.
            if "dlgagedeep" not in track.colnames:
                track.add_column(Column(dlgagedeep, "dlgagedeep"))
            else:
                track["dlgagedeep"] = dlgagedeep
        return

    def get_track(self, minit, feh):
        """ get the track closest to (minit, feh) """
        ind_mindist = np.argmin(
            (self.grid_minit - minit) ** 2. + (self.grid_feh - feh) ** 2.)
        return self.data[ind_mindist]

    def get_track_minit(self, minit):
        """ get the track closest to minit """
        chosen_minit = self.u_minit[np.argmin((self.u_minit - minit) ** 2)]
        ind_minit = self.grid_minit == chosen_minit
        return self.data[ind_minit]

    def get_track_feh(self, feh):
        """ get the track closest to feh """
        chosen_feh = self.u_feh[np.argmin((self.u_feh - feh) ** 2)]
        ind_feh = self.grid_feh == chosen_feh
        return self.data[ind_feh]

    def dtdeep(self):
        """ calculate dtdeep for each track """

        pass

    def lnprior(minit, feh, age):
        # 1. determine deep = dt deep/


        return 0