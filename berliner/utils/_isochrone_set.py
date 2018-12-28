import numpy as np
from regli.regli import bisect_interval
from regli import Regli
from astropy.table import Table, Column, vstack
from .isochrone_interp import isoc_interp, isoc_linterp


class IsochroneSet:
    def __init__(self, isoc_lgage, isoc_feh, isocs, model=None, Zsun=0.0152):
        """isocs should be a list of astropy.table.Table / Isochrone instances
        """
        # self.isocs = isocs
        self.grid_lgage = np.unique(isoc_lgage)
        self.grid_feh = np.unique(isoc_feh)
        self.r = Regli(self.grid_lgage, self.grid_feh)
        self.ind_dict = self.r.ind_dict

        # sort isocs
        print("@IsochroneSet: sorting isocs ...")
        isocs_sorted = []
        i_isocs = []
        lgage_sorted = []
        feh_sorted = []
        for _lgage, _feh in self.r.flats:
            d_isoc = np.abs(isoc_lgage - _lgage) + np.abs(isoc_feh - _feh)
            i_isoc = np.argmin(d_isoc)
            # print(i_isoc, d_isoc[i_isoc])
            i_isocs.append(i_isoc)
            isocs_sorted.append(isocs[i_isoc])
            lgage_sorted.append(isoc_lgage[i_isoc])
            feh_sorted.append(isoc_feh[i_isoc])

        self.lgage = np.array(lgage_sorted)
        self.feh = np.array(feh_sorted)
        self.set_isocs(np.array(isocs_sorted))

        self.Zsun = Zsun
        self.post_proc(model=model, Zsun=Zsun)

    def __repr__(self):
        s = ("==============================================================\n"
             "o IsochroneSet [N_isoc={}   N_tot={}]\n"
             "- age grid[{}]: {}\n"
             "- feh grid[{}]: {}\n"
             "==============================================================\n"
             "Column names:\n"
             "{}\n"
             "==============================================================\n"
             ).format(
            len(self.isocs), np.sum([len(_) for _ in self]),
            len(self.grid_lgage), self.grid_lgage,
            len(self.grid_feh), self.grid_feh,
            self.colnames)
        return s

    @property
    def colnames(self):
        return self.isocs[0].colnames

    def set_isocs(self, isocs):
        self.isocs = isocs

    def get_a_isoc(self, lgage, feh):
        """ return the closest isoc """
        d_isoc = np.abs(self.lgage - lgage) + np.abs(self.feh - feh)
        i_isoc = np.argmin(d_isoc)
        return self.isocs[i_isoc]

    def post_proc(self, model=None, Zsun=0.0152):
        """ post-processing of isochrone

        Parameters
        ----------
        model:
            {None | "parsec" | "mist"}

        """
        if model is None:
            pass

        elif model == "parsec":
            print("@post_proc: processing PARSEC isochrones ...")
            for isoc in self.isocs:
                _lgmini = Column(np.log10(isoc["M_ini"]), name="_lgmini")
                _fehini = Column(np.log10(isoc["Z"]/Zsun), name="_fehini") #Z or M ???
                _lgage  = Column(np.log10(isoc["log(age/yr)"]), name="_lgage")
                add_columns_safely(isoc, (_lgmini, _lgage, _fehini))
            self.model = model

        elif model == "mist":
            print("@post_proc: processing MIST isochrones ...")
            for isoc in self.isocs:
                _lgmini = Column(np.log10(isoc["initial_mass"]), name="_lgmini")
                _fehini = Column(isoc["Fe/H]_init"], name="_fehini")
                _lgage = Column(np.log10(isoc["log10_isochrone_age_yr"]), name="_lgage")
                add_columns_safely(isoc, (_lgmini, _lgage, _fehini))
            self.model = model

        else:
            raise ValueError("@Isochrone: model *{}* unknown!".format(model))

    def __getitem__(self, item):
        return self.isocs[item]

    def isoc_linterp(self, restrictions=(('logG', 0.01), ('logTe', 0.01)),
                     interp_colnames=('logG', 'logTe'), sampling_factor=1.0,
                     sampling_mode='linear', M_ini='M_ini',
                     n_jobs=-1, verbose=10):
        if interp_colnames == "all":
            interp_colnames = self.colnames

        from joblib import Parallel, delayed
        isoc_interp = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(isoc_linterp)(isoc, restrictions=restrictions,
                                  sampling_factor=sampling_factor,
                                  sampling_mode=sampling_mode,
                                  interp_colnames=interp_colnames,
                                  M_ini=M_ini) for isoc in self.isocs)

        return IsochroneSet(self.lgage, self.feh, isoc_interp)

    def vstack(self):
        return vstack(tuple(self.isocs))

    def make_prior(self):
        pass


def add_column_safely(t, col):
    if col.name in t.colnames:
        t[col.name] = col.data
    else:
        t.add_column(col)
    return


def add_columns_safely(t, cols):
    for col in cols:
        add_column_safely(t, col)
    return
