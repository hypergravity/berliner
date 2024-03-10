import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import Iterable, Optional


def generate_3darc(z=0.0, r=1.0, npt=100):
    x = np.zeros((npt, 3), dtype=float)
    x[:, 2] = z
    theta = np.linspace(0, np.pi / 2, npt)
    x[:, 0] = r * np.cos(theta)
    x[:, 1] = r * np.sin(theta)
    return x


def eval_curve_length(x: npt.NDArray, metric=None):
    """x is 2D array"""
    if metric is None:
        dx = np.diff(x, axis=0, prepend=[x[0]])
    else:
        dx = np.diff(x, axis=0, prepend=[x[0]]) * metric
    return np.cumsum(np.sqrt(np.sum(dx ** 2, axis=1)))


def interpolate_curve(
        xs: Iterable[npt.NDArray],
        ws: Iterable[npt.NDArray],
        metric: Optional[npt.NDArray] = None,
        n_interp=None,
) -> npt.NDArray:
    nx = len(xs)
    ndim = xs[0].shape[1]

    # evaluate normalized curve length
    curve_length = [eval_curve_length(xs[ix], metric) for ix in range(nx)]
    curve_length_max = [curve_length[ix][-1] for ix in range(nx)]
    normalized_curve_length = [
        curve_length[ix] / curve_length_max[ix] for ix in range(nx)
    ]

    # construct interpolation sample
    if n_interp is None:
        # combined_normalized_curve_length
        curve_length_interp = np.sort(np.hstack(normalized_curve_length))
    else:
        curve_length_interp = np.linspace(0, 1, n_interp)

    # interpolate curves
    xs_interp = np.concatenate(
        [
            np.hstack(
                [
                    np.interp(
                        curve_length_interp,
                        normalized_curve_length[ix],
                        xs[ix][:, idim],
                    )[:, None]
                    for idim in range(ndim)
                ]
            )[:, :, None]
            for ix in range(nx)
        ],
        axis=2,
    )
    # normalize weights
    nomalized_ws = np.asarray(ws) / np.sum(ws)
    # weighted average
    x_weighted = np.sum(xs_interp * nomalized_ws[None, None, :], axis=2)
    return x_weighted


def test_diff_prepend():
    x = np.linspace(0, 1.0, 100)
    y1 = x ** 0 * 1
    y2 = x ** 0 * 2
    # y1 = x**1+.5+10

    plt.plot(x, y1)
    # plt.plot(x, np.cumsum(y0))
    plt.plot(x, y2)
    f = 0.5
    prepend = f * y1[0] + (1 - f) * y2[0]
    # prepend -> the value is prepended before diff operation
    ymean = np.diff(f * np.cumsum(y1) + (1 - f) * np.cumsum(y2), prepend=[prepend])
    plt.plot(x, ymean)
    # plt.plot(x, np.cumsum(y1))
    # plt.plot(x, np.hstack([]))
