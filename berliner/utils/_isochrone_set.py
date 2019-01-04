import numpy as np
from regli.regli import bisect_interval
from regli import Regli
from astropy.table import Table, Column, vstack
from .isochrone_interp import isoc_interp, isoc_linterp


class IsochroneGrid:
    def __init__(self, isoc_lgage, isoc_fehini, isocs, model=None, Zsun=0.0152):
        """isocs should be a list of astropy.table.Table / Isochrone instances
        """
        # self.isocs = isocs
        self.grid_lgage = np.unique(isoc_lgage)
        self.grid_fehini = np.unique(isoc_fehini)
        self.r = Regli(self.grid_lgage, self.grid_fehini)
        self.ind_dict = self.r.ind_dict

        # sort isocs
        print("@IsochroneGrid: sorting isocs ...")
        isocs_sorted = []
        i_isocs = []
        lgage_sorted = []
        fehini = []
        for _lgage, _feh in self.r.flats:
            d_isoc = np.abs(isoc_lgage - _lgage) + np.abs(isoc_fehini - _feh)
            i_isoc = np.argmin(d_isoc)
            # print(i_isoc, d_isoc[i_isoc])
            i_isocs.append(i_isoc)
            isocs_sorted.append(isocs[i_isoc])
            lgage_sorted.append(isoc_lgage[i_isoc])
            fehini.append(isoc_fehini[i_isoc])

        self.lgage = np.array(lgage_sorted)
        self.fehini = np.array(fehini)
        self.set_isocs(np.array(isocs_sorted))

        self.Zsun = Zsun
        self.post_proc(model=model, Zsun=Zsun)

        self.delta_ready = False

    def __repr__(self):
        s = ("==============================================================\n"
             "* IsochroneGrid [N_isoc={}   N_tot={}]\n"
             "* age grid[{}]: {}\n"
             "* feh grid[{}]: {}\n"
             "==============================================================\n"
             "Column names:\n"
             "{}\n"
             "==============================================================\n"
             ).format(
            len(self.isocs), np.sum([len(_) for _ in self]),
            len(self.grid_lgage), self.grid_lgage,
            len(self.grid_fehini), self.grid_fehini,
            self.colnames)
        return s

    @staticmethod
    def add_column(tbl, col, name=None):
        if name is None:
            add_column_safely(tbl, col)
        else:
            add_column_safely(tbl, Column(col, name))
        return

    @property
    def colnames(self):
        return self.isocs[0].colnames

    def set_isocs(self, isocs):
        self.isocs = isocs

    def get_a_isoc(self, lgage, feh):
        """ return the closest isoc """
        d_isoc = np.abs(self.lgage - lgage) + np.abs(self.fehini - feh)
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

        elif model == "parsec12s":
            print("@post_proc: processing PARSEC isochrones ...")
            for isoc in self.isocs:
                _lgmini = Column(np.log10(isoc["M_ini"]), name="_lgmini")
                _fehini = Column(np.log10(isoc["Z"]/Zsun), name="_fehini") #Z or M ???
                _lgage  = Column(isoc["log(age/yr)"], name="_lgage")
                add_columns_safely(isoc, (_lgmini, _lgage, _fehini))
            self.model = model

        elif model == "mist":
            print("@post_proc: processing MIST isochrones ...")
            for isoc in self.isocs:
                _lgmini = Column(np.log10(isoc["initial_mass"]), name="_lgmini")
                _fehini = Column(isoc["Fe/H]_init"], name="_fehini")
                _lgage = Column(isoc["log10_isochrone_age_yr"], name="_lgage")
                add_columns_safely(isoc, (_lgmini, _lgage, _fehini))
            self.model = model

        else:
            raise ValueError("@Isochrone: model *{}* unknown!".format(model))

    def __getitem__(self, item):
        return self.isocs[item]

    def isoc_linterp(self, restrictions=(('logG', 0.01), ('logTe', 0.01)),
                     interp_colnames=('logG', 'logTe'), sampling_factor=1.0,
                     M_ini='M_ini', n_jobs=-1, verbose=10):
        if interp_colnames == "all":
            interp_colnames = self.colnames

        from joblib import Parallel, delayed
        isoc_interp_ = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(isoc_linterp)(isoc, restrictions=restrictions,
                                  sampling_factor=sampling_factor,
                                  # sampling_mode=sampling_mode,
                                  interp_colnames=interp_colnames,
                                  M_ini=M_ini) for isoc in self.isocs)

        return IsochroneGrid(self.lgage, self.fehini, isoc_interp_)

    def vstack(self):
        return vstack(tuple(self.isocs))

    def make_delta(self):
        # [d_lgage, d_fehini, d_mini]
        _d_lgage = make_delta(self.grid_lgage, "log10")
        _d_fehini = make_delta(self.grid_fehini, "linear")
        print("@make_delta: setting deltas for isochrones ...")
        for (i_lgage, i_fehini), i_isoc in self.r.ind_dict.items():
            n_pts = len(self.isocs[i_isoc])
            self.add_column(self.isocs[i_isoc],
                            np.ones((n_pts,), float) * _d_lgage[i_lgage],
                            "_d_age")
            self.add_column(self.isocs[i_isoc],
                            np.ones((n_pts,), float) * _d_fehini[i_fehini],
                            "_d_fehini")
            # self.add_column(self.isocs[i_isoc],
            #                 make_delta(self.isocs[i_isoc]["M_ini"], "linear"),
            #                 "_d_mini")
            self.add_column(self.isocs[i_isoc],
                            make_delta(self.isocs[i_isoc]["_lgmini"], "log10"),
                            "_d_mini")
        self.delta_ready = True

    def make_prior_sb(self, func_prior=None, qs=["logTe", "logG"]):
        if not self.delta_ready:
            raise ValueError("@IsochroneGrid: delta is not ready!")
        else:
            if func_prior is None:
                from berliner.utils.imf import imf

                def func_prior(feh, lgage, lgmini):
                    return imf(10 ** lgmini, kind="salpeter")

            # stack isochrones
            isoc_stacked = self.vstack()
            # evaluate function prior / [IMF in the simplest case]
            w = func_prior(isoc_stacked["_lgage"], isoc_stacked["_fehini"],
                           isoc_stacked["_lgmini"])
            # d_volume
            dv = isoc_stacked["_d_age"] * isoc_stacked["_d_fehini"] * \
                 isoc_stacked["_d_mini"]

            return np.array([isoc_stacked[q] for q in qs]).T, np.array(w*dv)

    # for histogram
    @staticmethod
    def edge_to_center(edge):
        return np.array(edge[:-1]+np.diff(edge))

    @staticmethod
    def edges_to_centers(edges):
        return [np.array(edge[:-1]+np.diff(edge)) for edge in edges]

    @staticmethod
    def center_to_edge(center):
        pass

    @staticmethod
    def centers_to_edges(centers):
        pass


# for table manipulations
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


# for isochrone delta
def make_delta(x, scale="linear", loose=True):
    if scale == "linear":
        if loose:
            xedges = np.concatenate(([1.5 * x[0] - 0.5 * x[1]],
                                     x[:-1] + np.diff(x) / 2.,
                                     [1.5 * x[-1] - 0.5 * x[-2]]))
        else:
            xedges = np.concatenate(
                ([x[0]], x[:-1] + np.diff(x) / 2., [x[-1]]))
        return np.diff(xedges)
    elif scale == "log10":
        x = 10 ** x
        if loose:
            xedges = np.concatenate(([1.5 * x[0] - 0.5 * x[1]],
                                     x[:-1] + np.diff(x) / 2.,
                                     [1.5 * x[-1] - 0.5 * x[-2]]))
        else:
            xedges = np.concatenate(
                ([x[0]], x[:-1] + np.diff(x) / 2., [x[-1]]))
        return np.diff(xedges)
