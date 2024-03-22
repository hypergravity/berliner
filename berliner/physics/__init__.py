import numpy as np

# ######################## #
# to evaluate [M/H] from Z #
# ######################## #


# parsec
def eval_mh_parsec(Z):
    ZXsun = 0.0207
    Y = 0.2485 + 1.78 * Z
    X = 1 - Y - Z
    MH = np.log10(Z / X) - np.log10(ZXsun)
    return MH


# mist
def eval_zx_mist(Z):
    Yp = 0.249
    Ypsun = 0.2703
    Zpsun = 0.0142
    Y = Yp + (Ypsun - Yp) / Zpsun * Z
    X = 1 - Y - Z
    return np.log10(Z / X)


def eval_mh_mist(Z):
    Zpsun = 0.0142
    return eval_zx_mist(Z) - eval_zx_mist(Zpsun)


# combined
def eval_mh(Z, model="parsec"):
    if model == "parsec":
        return eval_mh_parsec(Z)
    elif model == "mist":
        return eval_mh_mist(Z)
    else:
        raise ValueError("@IsochroneSet: invalid model!")


def eval_mbol(log_L):
    Mbol = 4.77 - 2.5 * log_L
    return Mbol


def eval_logg(log_mass, log_Teff, log_L):
    log_g = -10.616 + log_mass + 4.0 * log_Teff - log_L
    return log_g
