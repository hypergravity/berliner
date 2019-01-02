import numpy as np


def imf(m, kind="salpeter", **kwargs):
    if kind in {"salpeter", "miller", "kroupa"}:
        # valid kind
        if kind == "salpeter":
            return Salpeter1955(m, **kwargs)
        elif kind == "miller":
            return MillerScalo1979(m, **kwargs)
        elif kind == "kroupa":
            return Kroupa2001(m, **kwargs)

    elif kind in {"chabrier1", "chabrier2"}:
        # not implemented
        raise DeprecationWarning("Not implemented for kind = {}!".format(kind))

    else:
        # invalid kind
        raise ValueError("Invalid value for kind!")


def Salpeter1955(m, xi0=1., alpha=2.35):
    return xi0 * m ** (-alpha)


def MillerScalo1979(m, xi0=1., alphas=(2.35, 1)):
    return np.piecewise(m, [m>=0, m<1],
                        [xi0 * m ** -alphas[0],
                         xi0 * m ** -alphas[1]])


def Kroupa2001(m, xi0=1., alphas=(0.3, 1.3, 2.3)):
    return np.piecewise(m, [m < 0.08, (0.08 < m) & (m < 0.5), m > 0.5],
                        [xi0 * m ** (-alphas[0]),
                         xi0 * m ** (-alphas[1]),
                         xi0 * m ** (-alphas[2])])


def Chabrier03Individual(m):
    pass