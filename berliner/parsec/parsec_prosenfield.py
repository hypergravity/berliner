from astropy.table import Table
import os
import re
import numpy as np


def split_filename(fp):
    fn = os.path.basename(fp)
    Z, Y, OUTA, Minit = [np.float(_)
                   for _ in re.split(r"match_|[ZYM_]|F7|OUTA|.DAT|.dat|.HB|ADD", fn)
                   if _ is not ""]
    return Z, Y, OUTA, Minit


def read_parsec(fp):

    t = Table.read(fp, format="ascii")

    fn = os.path.basename(fp)
    Z, Y, OUTA, Minit = split_filename(fn)
    t.meta["Z"] = Z
    t.meta["Y"] = Y
    t.meta["OUTA"] = OUTA
    t.meta["MINIT"] = Minit
    t.meta["HB"] = "HB" in fn
    t.meta["ADD"] = "ADD" in fn

    return t


def read_rosenfield(fp):

    t = Table.read(fp, format="ascii.commented_header")

    fn = os.path.basename(fp)
    Z, Y, OUTA, Minit = split_filename(fn)
    t.meta["Z"] = Z
    t.meta["Y"] = Y
    t.meta["OUTA"] = OUTA
    t.meta["MINIT"] = Minit
    t.meta["HB"] = "HB" in fn
    t.meta["ADD"] = "ADD" in fn

    return t

