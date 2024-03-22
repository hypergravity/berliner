# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 20:01:38 2017

@author: cham
"""

import numpy as np
from astropy.table import Table, Column
from scipy.interpolate import interp1d


def isoc_interp(isoc,
                restrictions=(('logG', 0.01), ('logTe', 0.01)),
                sampling_factor=1.0, sampling_mode='linear',
                interp_config=(('logG', 'linear'), ('logTe', 'linear')),
                M_ini='M_ini'):
    """ isochrone interpolation that doesn't lose any structures

    Parameters
    ----------
    isoc: Table
        isochrone table
    restrictions: tuple
        (colname, maxstep) pairs
    sampling_factor: float
        sampling doubling factor
    sampling_mode: 'linear' | 'random'
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
        n_interp_points_ = np.ceil(np.abs(est_diff) / est_step * sampling_factor) + 1
        n_interp_points = np.where(n_interp_points > n_interp_points_,
                                   n_interp_points, n_interp_points_)
    n_interp_points[-1] += 1

    # sampling scheme {linear|random}
    if sampling_mode is 'linear':
        # linspace
        for i in range(n_row - 2):
            M_interp = np.hstack((M_interp, np.linspace(M[i], M[i + 1], n_interp_points[i])[:-1]))
        # last interval is special
        i = n_row - 2
        M_interp = np.hstack(
            (M_interp, np.linspace(M[i], M[i + 1], n_interp_points[i])))
    elif sampling_mode is 'random':
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


def isoc_linterp(isoc,
                 restrictions=(('logG', 0.01), ('logTe', 0.01)),
                 sampling_factor=1.0, sampling_mode='linear',
                 interp_colnames=('logG', 'logTe'),
                 Mini='M_ini'):
    """ isochrone interpolation that doesn't lose any structures

    Parameters
    ----------
    isoc: Table
        isochrone table
    restrictions: tuple
        (colname, maxstep) pairs
    sampling_factor: float
        sampling doubling factor
    sampling_mode: 'linear' | 'random'
        sampling scheme
    interp_config: tuple
        (colname, kind) pairs
    Mini: string
        the column name of the X in interpolation, default is 'M_ini'

    Returns
    -------
    Table of interpolated isochrone

    """

    n_row = len(isoc)

    # kick invalid values
    M = isoc[Mini].data
    M_diff = np.diff(M)

    # determine M_interp
    M_interp = np.zeros(0)

    n_interp_points = np.ones((n_row - 1,)) * 2
    for est_colname, est_step in restrictions:
        est_diff = np.diff(isoc[est_colname].data)
        # n_interp_points = np.max(np.array([n_interp_points, (
        #     np.ceil(np.abs(est_diff) / est_step)) * doubling + 1]), axis=0)
        n_interp_points_ = np.ceil(np.abs(est_diff) / est_step * sampling_factor) + 1
        n_interp_points = np.where(n_interp_points > n_interp_points_,
                                   n_interp_points, n_interp_points_)
    n_interp_points[-1] += 1

    # sampling scheme {linear|random}
    if sampling_mode == 'linear':
        # linspace
        for i in range(n_row - 2):
            M_interp = np.hstack((M_interp, np.linspace(M[i], M[i + 1], n_interp_points[i])[:-1]))
        # last interval is special
        i = n_row - 2
        M_interp = np.hstack(
            (M_interp, np.linspace(M[i], M[i + 1], n_interp_points[i])))
    elif sampling_mode == 'random':
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
        raise ValueError('@isoc_linterp: sampling_mode [{}] is not in (linear|random)!'.format(sampling_mode))

    # interpolation
    col_list = []
    for colname in interp_colnames:
        assert colname in isoc.colnames
        col_list.append(Column(
            interp1d(isoc[Mini], isoc[colname], kind="linear")(M_interp),
            colname))

    # return interpolated isochrone table
    return Table(col_list)