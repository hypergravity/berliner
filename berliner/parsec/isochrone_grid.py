# -*- coding: utf-8 -*-
"""

Author
------
Bo Zhang

Email
-----
bozhang@nao.cas.cn

Created on
----------
- Fri Mar  25 17:57:00 2016     get a set of PADOVA isochrones grid

Modifications
-------------
- Fri Mar  25 17:57:00 2016     get a set of PADOVA isochrones grid

Aims
----
- get a set of PADOVA isochrones grid
- output the combined isochrone table

"""

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from joblib import Parallel, delayed
from scipy.interpolate import PchipInterpolator

from . import ezpadova_wrapper

from .isochrone import Zsun, Zmin, Zmax, logtmax, logtmin


def __find_valid_grid(grid_feh, grid_logt, Zsun=Zsun):
    """ return a valid grid of [feh, logt]

    Parameters
    ----------
    grid_feh: array
        a list [Fe/H] values for isochrone grid
    grid_logt: array
        a list logt values for isochrone grid
    Zsun: float
        the solar metallicity

    """

    grid_feh = np.array(grid_feh).flatten()

    # grid
    grid_logt = np.array(grid_logt).flatten()
    grid_Z = 10. ** grid_feh * Zsun

    ind_valid_Z = np.logical_and(grid_Z >= Zmin, grid_Z <= Zmax)
    ind_valid_logt = np.logical_and(grid_logt >= logtmin, grid_logt <= logtmax)

    # valid grid
    vgrid_feh = grid_feh[ind_valid_Z]
    vgrid_logt = grid_logt[ind_valid_logt]

    # verbose
    print("==================================================================")
    print("@Cham: the valid range for Z & logt are (%s, %s) & (%s, %s)."
          % (Zmin, Zmax, logtmin, logtmax))
    print("------------------------------------------------------------------")
    print("@Cham: valid input feh are:  %s" % vgrid_feh)
    print("@Cham: valid input logt are: %s" % vgrid_logt)
    print("@Cham: INvalid input feh are: %s" % grid_feh[~ind_valid_Z])
    print("@Cham: INvalid input logt are: %s" % grid_logt[~ind_valid_logt])
    print("==================================================================")

    return vgrid_feh, vgrid_logt


def get_isochrone_grid(grid_feh, grid_logt, model="parsec12s", phot="sloan",
                       Zsun=Zsun, silent=True, n_jobs=8, verbose=10,
                       **kwargs):
    """ get a list of isochrones using EZPADOVA

    Parameters
    ----------
    grid_feh: array
        [Fe/H] grid
    grid_logt: array
        logt grid
    model: string
        default is "parsec12s"
    phot: string
        default is "sloan"
    Zsun: float
        default is 0.0152
    n_jobs: int
        if parflat is True, specify number of jobs in JOBLIB
    verbose: int/bool
        verbose level
    **kwargs:
        other keyword arguments for cmd.get_one_isochrone()

    Returns
    -------
    vgrid_feh, vgrid_logt, grid_list, isoc_list

    """
    # validate grid
    vgrid_feh, vgrid_logt = __find_valid_grid(grid_feh, grid_logt, Zsun=Zsun)

    # construct list
    grid_list = []
    isoc_lgage = []
    isoc_feh = []
    for grid_feh_ in vgrid_feh:
        for grid_logt_ in vgrid_logt:
            grid_list.append((10. ** grid_logt_, 10. ** grid_feh_ * Zsun))
            isoc_lgage.append(grid_logt_)
            isoc_feh.append(grid_feh_)
    isoc_lgage = np.array(isoc_lgage)
    isoc_feh = np.array(isoc_feh)

    print("@Cham: you have requested for %s isochrones!" % len(grid_list))
    print("==================================================================")

    # get isochrones
    if n_jobs > 1 or n_jobs == -1:
        # get isochrones in parallel
        isoc_list = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(ezpadova_wrapper.get_one_isochrone_silently)(
                grid_list_[0], grid_list_[1], silent=silent, model=model,
                phot=phot, **kwargs)
            for grid_list_ in grid_list)
    else:
        # get isochrones sequentially
        isoc_list = []
        for i, grid_list_ in enumerate(grid_list):
            print(("@Cham: sending request for isochrone "
                   "| (logt=%s, [Fe/H]=%s) | (t=%s, Z=%s) | [%s/%s]...")
                  % (np.log10(grid_list_[0]), np.log10(grid_list_[1]/Zsun),
                     grid_list_[0], grid_list_[1],
                     i+1, len(grid_list)))
            isoc_list.append(ezpadova_wrapper.get_one_isochrone_silently(
                grid_list_[0], grid_list_[1], model=model, phot=phot,
                silent=silent, **kwargs))

    # verbose
    print("@Cham: got all requested isochrones!")
    print("==================================================================")
    print("@Cham: colnames are:")
    print(isoc_list[0].colnames)
    print("==================================================================")

    # return vgrid_feh, vgrid_logt, grid_list, isoc_list
    return np.array(isoc_lgage), np.array(isoc_feh), isoc_list


# useless & deprecated
def interpolate_to_cube(grid_feh, grid_logt, grid_mini, isoc_list,
                        cube_quantities=[]):
    """ interpolate a slit of isochrones into data cubes
    grid_feh: array
        [Fe/H] grid
    grid_logt: array
        logt grid
    grid_mini: array
        the M_ini array to which interpolate into
    isoc_list: list
        a list of isochrones (in astropy.table.Table form)
    grid_list: list
        a list of (logt, Z) tuples corresponding to isoc_list
    cube_quantities: list
        a list of names of the quantities to be interpolated

    Returns
    -------
    cube_data_list, cube_name_list

    """
    # flatten grid
    grid_feh = np.array(grid_feh).flatten()
    grid_logt = np.array(grid_logt).flatten()
    grid_mini = np.array(grid_mini).flatten()

    # mesh cube
    cube_logt, cube_feh, cube_mini, = np.meshgrid(grid_logt, grid_feh, grid_mini)
    cube_size = cube_feh.shape
    print("@Cham: cube shape: ", cube_size)
    print("@Cham: -------------------------------------------------------")

    # determine cube-quantities
    if len(cube_quantities) == 0:
        # all the quantities besides [feh, logt, Mini]
        colnames = list(isoc_list[0].colnames)
        assert colnames[0] == "Z"
        assert colnames[1] == "logageyr"
        assert colnames[2] == "M_ini"
        cube_quantities = colnames[3:]
    print("@Cham: Interpolating these quantities into cubes ...")
    print("%s" % cube_quantities)
    print("@Cham: -------------------------------------------------------")

    # smoothing along M_ini
    for i in range(len(isoc_list)):
        # Tablize
        if not isinstance(isoc_list[i], Table):
            isoc_list[i] = Table(isoc_list[i].data)
        # smoothing M_ini
        ind_same_mini = np.hstack((False, np.diff(isoc_list[i]["M_ini"].data)==0))
        sub_same_mini = np.arange(len(isoc_list[i]))[ind_same_mini]
        isoc_list[i].remove_rows(sub_same_mini)

        print("@Cham: smoothing isochrones [%s/%s] | %s rows removed ..."
              % (i + 1, len(isoc_list), len(sub_same_mini)))
    print("@Cham: -------------------------------------------------------")

    # interpolation
    cube_data_list = [cube_feh, cube_logt, cube_mini]
    cube_name_list = ["feh", "logt", "M_ini"]
    for k in range(len(cube_quantities)):
        cube_name = cube_quantities[k]
        c = 0
        cube_data = np.ones(cube_size) * np.nan
        for i in range(len(grid_feh)):
            for j in range(len(grid_logt)):
                this_isoc = isoc_list[c]
                P = PchipInterpolator(this_isoc["M_ini"].data,
                                      this_isoc[cube_name].data,
                                      extrapolate=False)
                # return NaNs when extrapolate
                cube_data[i, j, :] = P(grid_mini)
                print("@Cham: Interpolating [%s] | {quantity: %s/%s} (%s/%s) ..."
                      % (cube_name, k + 1, len(cube_quantities), c + 1,
                         len(grid_feh) * len(grid_logt)))
                c += 1
        cube_data_list.append(cube_data)
        cube_name_list.append(cube_name)
    print("@Cham: -------------------------------------------------------")

    return cube_data_list, cube_name_list


def cubelist_to_hdulist(cube_data_list, cube_name_list):
    """ transform data cubes into fits HDU list

    Parameters
    ----------
    cube_data_list: list
        a list of cube data
    cube_name_list: list
        a list of quantity names for cube data

    """
    print("@Cham: transforming data cubes into HDU list ...")

    # construct Primary header
    header = fits.Header()
    header["author"] = "Bo Zhang (@NAOC)"
    header["data"] = "isochrone cube"
    header["software"] = "cube constructed using BOPY.HELPER.EZPADOVA"

    # initialize HDU list
    hl = [fits.hdu.PrimaryHDU(header=header)]

    # construct HDU list
    for i in range(len(cube_data_list)):
        hl.append(
            fits.hdu.ImageHDU(data=cube_data_list[i], name=cube_name_list[i]))

    print("@Cham: -------------------------------------------------------")
    return fits.HDUList(hl)


def combine_isochrones(isoc_list):
    """ combine a list of isochrone Tables into 1 Table

    Parameters
    ----------
    isoc_list: list
        a list of isochrones (astropy.table.Table format)

    """

    if isinstance(isoc_list[0], Table):
        # assume that these data are all Table
        comb_isoc = vstack(isoc_list)
    else:
        # else convert to Table
        for i in range(isoc_list):
            isoc_list[i] = Table(isoc_list[i])
        comb_isoc = vstack(isoc_list)

    return comb_isoc


def write_isoc_list(isoc_list, grid_list,
                    dirpath="comb_isoc_parsec12s_sloan", extname=".fits",
                    Zsun=0.0152):
    """ write isochrone list into separate tables

    Parameters
    ----------
    isoc_list: list
        a list of isochrones (in astropy.table.Table format)
    grid_list: list
        (10.**grid_logt_, 10.**grid_feh_*Zsun) pairs
    dirpath: string
        the directory path
    extname: string
        the ext name
    Zsun: float
        the solar metallicity

    """

    assert len(isoc_list) == len(grid_list)
    for i in range(len(isoc_list)):
        fp = dirpath + \
            "_ZSUN" + ("%.5f" % Zsun).zfill(7) + \
            "_LOGT" + ("%.3f" % np.log10(grid_list[i][0])).zfill(6) + \
            "_FEH" + ("%.3f" % np.log10(grid_list[i][1]/Zsun)).zfill(6) + \
            extname
        print("@Cham: writing table [%s] [%s/%s]..." % (fp, i+1, len(isoc_list)))
        isoc_list[i].write(fp, overwrite=True)
    return


def _test():
    """ download a random set of isochrones (sloan)
    Examples
    --------
    >>> from berliner.parsec.isochrone_grid import \
    >>>     (get_isochrone_grid, interpolate_to_cube, cubelist_to_hdulist,
    >>>      combine_isochrones, write_isoc_list)
    """
    # set grid
    grid_logt = [6, 7., 9]
    grid_feh = [-2.2, -1., 0, 1., 10]
    grid_mini = np.arange(0.01, 12, 0.01)

    # get isochrones
    vgrid_feh, vgrid_logt, grid_list, isoc_list = get_isochrone_grid(
        grid_feh, grid_logt, model="parsec12s", phot="sloan", n_jobs=1)

    # transform into cube data
    cube_data_list, cube_name_list = interpolate_to_cube(
        vgrid_feh, vgrid_logt, grid_mini, isoc_list,
        cube_quantities=["M_act", "g", "r"])

    # cube HDUs
    hl = cubelist_to_hdulist(cube_data_list, cube_name_list)
    hl.info()
    # hl.writeto()

    # combine isochrone tables
    comb_isoc = combine_isochrones(isoc_list)
    # comb_isoc.write()

    # write isochrone list into separate files
    # write_isoc_list(isoc_list, grid_list, "/pool/comb_isoc")
    return hl


def _test2():
    """ download full set of isochrones (2MASS)
    Examples
    --------
    >>> from berliner.parsec.isochrone_grid import \
    >>>     (get_isochrone_grid, interpolate_to_cube, cubelist_to_hdulist,
    >>>      combine_isochrones, write_isoc_list)
    """

    # set grid
    grid_logt = np.arange(6.0, 10.5, 0.01)
    grid_feh = np.arange(-4., +1., 0.05)
    grid_mini = np.arange(0.01, 12, 0.01)

    # get isochrones
    vgrid_feh, vgrid_logt, grid_list, isoc_list = get_isochrone_grid(
        grid_feh, grid_logt, model="parsec12s", phot="2mass", n_jobs=100)

    # transform into cube data
    cube_data_list, cube_name_list = interpolate_to_cube(
        vgrid_feh, vgrid_logt, grid_mini, isoc_list,
        cube_quantities=["Z", "logageyr", "M_act", "logLLo", "logTe",
                         "logG", "mbol", "J", "H", "Ks", "int_IMF", "stage"])

    # cube HDUs
    hl = cubelist_to_hdulist(cube_data_list, cube_name_list)
    hl.info()
    hl.writeto("/pool/model/parsec/isocgrid/cube_isoc_2mass_full.fits",
               clobber=True)

    # combine isochrone tables
    comb_isoc = combine_isochrones(isoc_list)
    comb_isoc.write("/pool/model/parsec/isocgrid/comb_isoc_2mass_full.fits")

    # write isochrone list into separate files
    write_isoc_list(isoc_list, grid_list, "/pool/model/parsec/isocgrid/2mass/2mass")
    return hl


if __name__ == "__main__":
    _test()