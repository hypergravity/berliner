# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 21:57:16 2017

@author: cham

Aim: wrap astropy.table.Table as parsec class

"""

import numpy as np
from astropy.table import Table, vstack, Column
from scipy.interpolate import Rbf, LinearNDInterpolator
from tqdm import tqdm
from collections import OrderedDict


class Isochrone(Table):

    # old parameters, designed for scipy.interpolate.Rbf(), deprecated
    __interp_xks = ("M_ini", "logageyr", "feh")
    __interp_yks = ("logTe", "logG", "feh")
    __interp_xdim = 3
    __interp_ydim = 3
    __interp_eps = (0.3, 0.2, 0.2)
    __interp_function = "linear"
    __interp_kwargs = {}

    # new parameters, designed for scipy.interpolate.LNDI()
    interp_method = "linear"
    interp_fill_value = np.nan
    interp_rescale = False

    lndis = {}

    def cut_bounds(self, key, bds=(None, None)):
        assert key in self.colnames
        ind = (self[key] > bds[0]) & (self[key] < bds[1])
        if np.sum(ind) <= 0:
            Warning("@Cham: Found nothing! Check bounds!")
            return None
        else:
            return self[ind]

    def lndi_init(self, xks=("M_ini", "logageyr", "feh"),
                  yks=("logTe", "logG", "feh"), **kwargs):
        # clear lndis
        self.lndis = OrderedDict()

        # make array
        xks = np.array(xks)
        yks = np.array(yks)

        # assert xks in colnames
        for xk in xks:
            try:
                assert xk in self.colnames
            except AssertionError:
                raise(AssertionError(
                    "@Cham: '{}' is not in colnames!".format(xk)))

        # generate position array
        X = np.array([self[_].data for _ in xks]).T

        # setup lndi or setup return columns
        # LinearNDInterpolator(points,values,fill_value=np.nan,rescale=False)
        for yk in yks:
            if yk in xks:
                # directly return input
                print("@parsec.lndi: {} found in input keys ...".format(yk))
                self.lndis[yk] = np.int(np.where(xks == yk)[0])
            else:
                # make lndi
                print("@parsec.lndi: setting up lndi for {} ...".format(yk))
                self.lndis[yk] = LinearNDInterpolator(
                    X, self[yk].data, **kwargs)
        print("@parsec.lndi: lndi setup successfully ...\n")
        return

    def lndi(self, Xp):
        results = []

        # do interpolation
        for lndi_ in self.lndis.values():
            if isinstance(lndi_, int):
                # int, return directly
                results.append(Xp[:, lndi_])
            elif isinstance(lndi_, LinearNDInterpolator):
                # lndi, interpolate
                results.append(lndi_(Xp))
            else:
                raise ValueError("lndis must be int or LNDI instance!")
        results = np.array(results).T

        # fill nan
        ind_inf = np.any(np.logical_not(np.isfinite(results)), axis=1)
        results[ind_inf] = np.nan

        return results

    def __init__(self, *args, model="parsec", Zsun=0.0152, mini="M_ini",
                 logt="logageyr", feh="feh", **kwargs):
        super().__init__(*args, **kwargs)
        # temporarily set basic parameters
        self.model = model
        self.Zsun = Zsun
        self.mini = mini
        self.logt = logt
        self.feh = feh
        # if feh == "fromZ":
        #     self.add_column(Column(np.log10(self["Z"]/Zsun), "feh"))
        # elif feh == "feh":
        #     self.feh = "feh"
        # else:
        #     raise(AssertionError("@Cham: which column is [M/H]?"))

    def set_basic_params(self, model="parsec", Zsun=0.0152,
                         mini="M_ini", logt="logageyr", feh="feh",):
        """ to set basic parameters of isochrone

        Parameters
        ----------
        model:
            the isochrone model name
        Zsun:
            the solar metallicity
        mini:
            the initial mass column name
        logt:
            the log10(age) column name
        feh:
            the [M/H] column name

        """

        self.model = model.upper()
        self.Zsun = Zsun

        try:
            assert mini in self.colnames
            self.mini = mini
        except AssertionError:
            raise AssertionError("@Cham: column '{}' not found!".format(mini))

        try:
            assert logt in self.colnames
            self.logt = logt
        except AssertionError:
            raise AssertionError("@Cham: column '{}' not found!".format(logt))

        try:
            assert feh in self.colnames
            self.feh = feh
        except AssertionError:
            raise AssertionError("@Cham: column '{}' not found!".format(feh))

        print("\n@Cham: basic parameters set successfully!")
        print("Model: {}".format(self.model))
        print("Zsun:  {}".format(self.Zsun))
        print("M_ini: --> '{}'".format(self.mini))
        print("logt:  --> '{}'".format(self.logt))
        print("[M/H]: --> '{}'".format(self.feh))

    def __repr__(self):
        # string for Z
        # if np.max(self['Z']) == np.min(self['Z']):
        #     s_z = "Z={:.7f}([M/H]={:.2f})".format(
        #         self['Z'][0], np.log10(self['Z'][0]/0.0152))
        #     c_z = 1
        # else:
        #     s_z = "Z=[{:.7f}, {:.7f}]([M/H]=[{:.2f}, {:.2f}])".format(
        #         np.min(self['Z']), np.max(self['Z']),
        #         np.log10(np.min(self['Z']) / 0.0152),
        #         np.log10(np.max(self['Z']) / 0.0152))
        #     c_z = len(np.unique(self['Z']))

        # string for [Fe/H]
        feh = self.feh
        if feh in self.colnames:
            if np.max(self[feh]) == np.min(self[feh]):
                s_z = "[M/H]={:.2f}".format(self[feh][0])
                c_z = 1
            else:
                s_z = "[M/H]=[{:.2f}, {:.2f}]".format(
                    np.min(self[feh]), np.max(self[feh]))
                c_z = len(np.unique(self[feh]))
        else:
            Warning("@Cham: there is no column '{}'".format(feh))
            s_z = "([M/H] column name not set correctly!)"
            c_z = np.nan

        # string for logt
        logt = self.logt
        if logt in self.colnames:
            if np.max(self[logt]) == np.min(self[logt]):
                s_t = "logt={:.2f}".format(self[logt][0])
                c_t = 1
            else:
                s_t = "logt=[{:.2f}, {:.2f}]".format(
                    np.min(self[logt]), np.max(self[logt]))
                c_t = len(np.unique(self[logt]))
        else:
            Warning("@Cham: there is no column '{}'".format(feh))
            s_t = "(logt column name not set correctly!)"
            c_t = np.nan

        # string for M_ini
        mini = self.mini
        if mini in self.colnames:
            s_m = "M_ini=[{:.3f}, {:.3f}]".format(
                np.min(self[mini]), np.max(self[mini]))
        else:
            s_m = "(M_ini column name not set correctly!)"

        # string for length
        s_l = "length={}".format(len(self))

        # grid?
        if c_z * c_t > 1:
            s_grid = " Grid: ({:d} [M/H] x {:d} logt)".format(c_z, c_t)
        else:
            s_grid = ""

        s = "<{} parsec{} {} {} {} {}>".format(
            self.model, s_grid, s_z, s_t, s_m, s_l)

        return s

    @staticmethod
    def join(isoc_list):

        # assert all isochrone tables have the same colnames
        colnames = isoc_list[0].colnames
        for i in range(len(isoc_list)):
            try:
                assert isoc_list[i].colnames == colnames
            except AssertionError as ae:
                print("parsec[0].colnames = ", colnames)
                print("parsec[{}].colnames = ", isoc_list[i].colnames)
                raise(ae)

        # try to join all isochrone tables
        return vstack([Isochrone(_) for _ in isoc_list])

    def interp_fakehw(self, X, xks=("M_ini", "logageyr", "feh"),
                      yks=("logTe", "logG", "feh"), eps_tau=0.2, eps_feh=0.2,
                      **kwargs):
        # 1. select neighboring isochrones accroding to tau & feh
        xks = np.array(xks)
        yks = np.array(yks)

        u_logt = np.unique(self[xks[1]])
        u_feh = np.unique(self[xks[2]])

        lxks = len(xks)
        lyks = len(yks)
        lX = len(X)

        Y = []
        for iX, X_ in enumerate(X):
            print("parsec.interp_fakehw: {}/{} .".format(iX+1, lX), end="")
            Y_ = np.zeros((lyks,), float)
            # select neighboring isochron data
            ind = (self[xks[1]] < X_[1] + eps_tau) & \
                  (self[xks[1]] >= X_[1] - eps_tau) & \
                  (self[xks[2]] < X_[2] + eps_feh) & \
                  (self[xks[2]] >= X_[2] - eps_feh)
            # if there is no neighboring isochrones
            try:
                assert np.sum(ind) > 0
            except AssertionError as ae:
                Y_ *= np.nan
                Y.append(Y_)
                print("", end="\n")
                continue
            # else,
            # generate position array
            Xdata = np.array([self[_][ind].data for _ in xks]).T
            for iyk, yk in enumerate(yks):
                if yk in xks:
                    ind_xk = np.where(xks == yk)[0]
                    Y_[iyk] = X_[ind_xk]
                    print(".", end="")
                else:
                    Y_[iyk] = LinearNDInterpolator(
                        Xdata, self[yk][ind].data)([X_], **kwargs)
                    print(".", end="")

            Y.append(Y_)
            print("", end="\n")
        return np.array(Y)

    def interp_init(self, xks=("M_ini", "logageyr", "feh"),
                    yks=("logTe", "logG", "feh"), **kwargs):
        # to initialize interpolator


        pass

    def interp(self, X):
        """

        Parameters
        ----------
        X:
            of shape (n_data, n_dim), the position of the interpolation

        Returns
        -------
        Y:
            of shape (n_data, n_dim), the position of the interpolation
        """

    def __interp_set(self, xks=("M_ini", "logageyr", "feh"),
                   yks=("logTe", "logG", "feh"), eps=(0.3, 0.2, 0.2),
                   function="linear", **kwargs):
        """ set the interpolation parameters

        :param xks:
            X coordinate keys
        :param yks:
            Y coordinate keys
        :param eps:
            the neighboring box
        :param function:
            interpolation function, conf scipy.interpolate.Rbf
        :param kwargs:
            will be passed to scipy.interpolate.Rbf
        """

        self.__interp_xks = xks
        self.__interp_yks = yks
        self.__interp_eps = eps
        self.__interp_function = function
        self.__interp_kwargs = kwargs

        self.__interp_xdim = len(xks)
        self.__interp_ydim = len(yks)

        return

    def __interp(self, Xs):
        """

        :param Xs:
            the X coordinates of interpolated positions
        :return:
        """
        try:
            assert Xs.ndim == 2
        except AssertionError as ae:
            print("@Cham: the dimension of input X must be 2!")
            raise(ae)

        result = []
        for i in tqdm(range(Xs.shape[0])):
            X = Xs[i]
            result.append(self.__interp_one(X))

        return np.array(result)

    def __interp_one(self, X):
        """ interpolate isochrone using scipy.interpolate.Rbf

        :param xs:
            X coordinates of the interpolated position, shape: (ndim,)

        :return:

        """
        try:
            assert X.ndim == 1
        except AssertionError as ae:
            print("@Cham: the dimension of input X must be 1!")
            raise (ae)

        try:
            X_upper = X + self.__interp_eps
            X_lower = X - self.__interp_eps

            ind = np.ones((len(self),), dtype=bool)
            for ixdim in range(self.__interp_xdim):
                ind &= (self[self.__interp_xks[ixdim]].data < X_upper[ixdim])
                ind &= (self[self.__interp_xks[ixdim]].data >= X_lower[ixdim])

            X_nb = np.array(self[self.__interp_xks][ind].to_pandas())
            Y_nb = np.array(self[self.__interp_yks][ind].to_pandas())

            # if neighbors too few, return nan
            if X_nb.shape[0] < self.__interp_xdim:
                print("\nWarning: for X = ", X, "returned nan, REASON: too few neighbors")
                return np.ones((self.__interp_ydim,), dtype=float) * np.nan

            result = list()
            # interpolate necessary columns
            for iyk, yk in enumerate(self.__interp_yks):
                if yk in self.__interp_xks:
                    # Y key in X keys, directly return it
                    # print("{} is found in XKS".format(yk),
                    #       np.where(np.array(self.interp_xks) == yk)[0])
                    result.append(X[np.where(np.array(self.__interp_xks) == yk)[0]])
                else:
                    # interpolate it
                    # print(X_nb.shape, Y_nb.shape)
                    rbfi = Rbf(*X_nb.T, Y_nb[:, iyk],
                               function=self.__interp_function, **self.__interp_kwargs)
                    result.append(rbfi(*X))
        except Exception as ee:
            print("\nWarning: for X = ", X, "returned nan, REASON: not sure...")
            result = np.ones((self.__interp_ydim,), dtype=float) * np.nan

        return np.array(result)

    def __interp_test(self, nstep=10):
        result = self.__interp(
            np.array(self[self.__interp_xks][::nstep].to_pandas()))
        return result
