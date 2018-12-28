from collections import OrderedDict

import numpy as np
from astropy.table import Table, vstack, Column


Zsun = 0.0152  # this value from CMD website
Zmin = 0.0001
Zmax = 0.07
logtmax = 10.13
logtmin = 1.


class Isochrone(Table):
    """ This class accepts astropy.table.Table objects with 4 columns:
            _lgmini:    log10(M_ini/M_sun)
            _feh:       [Fe/H]
            _lgage:     log10(age/yr)
            _eep:       EEP number

    """

    meta = OrderedDict()
    model = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_column_safely(self, col):
        if col.name in self.colnames:
            self[col.name] = col.data
        else:
            self.add_column(col)

    def post_proc(self, model=None):
        """ post-processing of isochrone

        Parameters
        ----------
        model:
            {None | "parsec" | "mist"}

        """
        if model is None:
            pass

        elif model == "parsec":
            _lgmini = Column(np.log10(self["Mini"]), name="_lgmini")
            _fehini = Column(np.log10(self["Zini"]/Zsun), name="_fehini") #Z or M ???
            _lgage  = Column(np.log10(self["Age"]), name="_lgage")
            for col in [_lgmini, _lgage, _fehini]:
                self.add_column_safely(col)
            self.model = model

        elif model == "mist":
            _lgmini = Column(np.log10(self["initial_mass"]), name="_lgmini")
            _fehini = Column(self["Fe/H]_init"], name="_fehini")
            _lgage = Column(np.log10(self["log10_isochrone_age_yr"]), name="_lgage")
            for col in [_lgmini, _lgage, _fehini]:
                self.add_column_safely(col)
            self.model = model

        else:
            raise ValueError("@Isochrone: model *{}* unknown!".format(model))

    @staticmethod
    def vstack(*args, **kwargs):
        return vstack(*args, **kwargs)

    @staticmethod
    def from_table(tbl):
        return Isochrone(tbl.columns, meta=tbl.meta)

    def to_table(self):
        return Table(self.columns)

    # info of isochrone
    @property
    def Neep(self):
        # N_points of isochrone
        return len(self)

    @property
    def Ncol(self):
        # N_columns of isochrone
        return len(self.colnames)

    # parameters of this isochrone
    @property
    def lgage(self):
        self._age = self["_lgage"].data[0]
        return self._age

    @property
    def feh(self):
        return self["_feh"].data[0]

    @property
    def maxlgmass(self):
        return np.max(self["_lgmass"])

    @property
    def minlgmass(self):
        return np.min(self["_lgmass"])

    @property
    def maxeep(self):
        return np.max(self["_eep"])

    @property
    def mineep(self):
        return np.min(self["_eep"])

    @staticmethod
    def combine(isolists):
        result = []
        for isolist in isolists:
            for iso in isolist:
                result.append(iso)
        return result