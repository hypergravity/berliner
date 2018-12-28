# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 20:01:38 2017

@author: cham
"""

import numpy as np
from astropy.table import Table, Column
from scipy.interpolate import interp1d


def isoc_interp(isoc,
                restrictions=(('logG', 0.01), ('M_act', 0.01)),
                doubling=1.0,
                mode='linear',
                interp_config=(('logG', 'linear'), ('J', 'linear')),
                M_ini='M_ini'):
    """ isochrone interpolation that doesn't lose any structures

    Parameters
    ----------
    isoc: Table
        isochrone table
    restrictions: tuple
        (colname, maxstep) pairs
    doubling: float
        simpling doubling factor
    mode: 'linear' | 'random'
        sampling scheme
    interp_config: tuple
        (colname, kind) pairs
    M_ini: string
        the column name of the X in interpolation, default is 'M_ini'

    Returns
    -------
    Table of interpolated isochrone

    """

    n_row = len(isoc)

    # kick invalid values
    M = isoc[M_ini].data
    M_diff = np.diff(M)

    # determine M_interp
    M_interp = np.zeros(0)

    n_interp_points = np.ones((n_row - 1,)) * 2
    for est_colname, est_step in restrictions:
        est_diff = np.diff(isoc[est_colname].data)
        # n_interp_points = np.max(np.array([n_interp_points, (
        #     np.ceil(np.abs(est_diff) / est_step)) * doubling + 1]), axis=0)
        n_interp_points_ = np.ceil(np.abs(est_diff) / est_step * doubling) + 1
        n_interp_points = np.where(n_interp_points > n_interp_points_,
                                   n_interp_points, n_interp_points_)
    n_interp_points[-1] += 1

    # sampling scheme {linear|random}
    if mode is 'linear':
        # linspace
        for i in range(n_row - 2):
            M_interp = np.hstack((M_interp, np.linspace(M[i], M[i + 1], n_interp_points[i])[:-1]))
        # last interval is special
        i = n_row - 2
        M_interp = np.hstack(
            (M_interp, np.linspace(M[i], M[i + 1], n_interp_points[i])))
    elif mode is 'random':
        # random
        for i in range(n_row - 2):
            M_interp = np.hstack(
                (M_interp,
                 np.random.rand(n_interp_points[i] - 1,) * M_diff[i] + M[i]))
        # last interval is special
        i = n_row - 2
        M_interp = np.hstack(
            (M_interp, np.linspace(M[i], M[i + 1], n_interp_points[i])))
    else:
        raise ValueError('mode is not in {linear|random}!')

    # interpolation
    col_list = []
    for interp_col, kind in interp_config:
        assert interp_col in isoc.colnames
        col_list.append(Column(interp1d(isoc[M_ini], isoc[interp_col], kind=kind)(M_interp), interp_col))

    # return interpolated isochrone table
    return Table(col_list)
