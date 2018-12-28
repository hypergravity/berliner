# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:22:24 2017

@author: cham

@aim:

To wrap ezpadova.

"""

import os
import sys
from ezpadova.parsec import get_one_isochrone, get_photometry_list, \
    get_t_isochrones, get_Z_isochrones
from ezpadova.simpletable import SimpleTable
from .isochrone import Isochrone
from astropy.table import Table
from collections import OrderedDict


__all__ = ["get_one_isochrone", "get_one_isochrone_silently",
           "get_photometry_list", "get_t_isochrones", "get_Z_isochrones"]


def get_one_isochrone_silently(*args, silent=True, **kwargs):
    """ To make the download silent.

    Parameters
    ----------

    age: float
        age of the isochrone (in yr)
    metal: float
        metalicity of the isochrone

    ret_table: bool
        if set, return a eztable.Table object of the data

    model: str
        select the type of model :func:`help_models`

    carbon: str
        carbon stars model :func:`help_carbon_stars`

    interp: str
        interpolation scheme

    dust: str
        circumstellar dust prescription :func:`help_circumdust`

    Mstars: str
        dust on M stars :func:`help_circumdust`

    Cstars: str
        dust on C stars :func:`help_circumdust`

    phot: str
        photometric set for photometry values :func:`help_phot`

    Returns
    -------
    r: Table or str
        if ret_table is set, return a eztable.Table object of the data
        else return the string content of the data

    """
    if silent:
        # silent download
        sys.stdout = open(os.devnull, "w")
        r = get_one_isochrone(*args, **kwargs)
        sys.stdout = sys.__stdout__

    else:
        r = get_one_isochrone(*args, **kwargs)

    # convert to .isochrone.Isochrone type
    try:
        return Table(r.data)
    except Exception:
        return Table(r.columns, meta=r.meta)
