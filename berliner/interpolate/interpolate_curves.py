import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import Iterable, Optional


def generate_3darc(z: float = 0.0, r: float = 1.0, n_pnt: int = 100):
    """Generate 3D arc."""
    x = np.zeros((n_pnt, 3), dtype=float)
    x[:, 2] = z
    theta = np.linspace(0, np.pi / 2, n_pnt)
    x[:, 0] = r * np.cos(theta)
    x[:, 1] = r * np.sin(theta)
    return x


def eval_curve_length(x: npt.NDArray, metric: Optional[npt.ArrayLike] = None):
    """Evaluate curve length for each chunk."""
    if metric is None:
        dx = np.diff(x, axis=0, prepend=[x[0]])
    else:
        dx = np.diff(x, axis=0, prepend=[x[0]]) * metric
    return np.sqrt(np.sum(dx**2, axis=1))


def eval_cumulated_curve_length(x: npt.NDArray, metric=None):
    """Evaluate cumulated curve length."""
    if metric is None:
        dx = np.diff(x, axis=0, prepend=[x[0]])
    else:
        dx = np.diff(x, axis=0, prepend=[x[0]]) * metric
    return np.cumsum(np.sqrt(np.sum(dx**2, axis=1)))


def angle_to_weight(angle: float, base: float = 10.0) -> float:
    """Convert angle (radian) to weight.
    Power-law is used to let weights dominated by high-curvature points.
    """
    return base**angle


def eval_angle(vec1, vec2, fill_value=0.0):
    """Evaluate the length of the vectors"""
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    # Calculate the dot product
    dot_product = np.dot(vec1, vec2)
    # Calculate the cosine of the angle
    cos_theta = dot_product / (norm_a * norm_b)
    # Calculate the angle
    theta = np.arccos(cos_theta)
    return theta if np.isfinite(theta) else fill_value


def eval_angle_weighted_cumulated_curve_length(
    x: npt.NDArray,
    metric: Optional[npt.ArrayLike] = None,
    angle_weight_base: float = 10.0,
) -> npt.NDArray:
    """Evaluate angle-weighted cumulated curve length."""
    n_pnt = x.shape[0]
    # calculate difference
    dx_minus = np.diff(x, prepend=[x[0]], axis=0)
    dx_plus = np.diff(x, append=[x[-1]], axis=0)
    # calculate angle
    angle = np.array([eval_angle(dx_minus[i], dx_plus[i]) for i in range(n_pnt)])
    # calculate weight
    angle_weight = angle_to_weight(angle, base=angle_weight_base)
    # calculate curve length
    curve_length = eval_curve_length(x, metric=metric)
    # calculate angle-weighted curve length
    return np.cumsum(angle_weight * curve_length)


def interpolate_curve(
    xs: Iterable[npt.NDArray],
    ws: Iterable[npt.NDArray],
    metric: Optional[npt.NDArray] = None,
    n_interp=None,
) -> npt.NDArray:
    nx = len(xs)
    ndim = xs[0].shape[1]

    # evaluate normalized curve length
    curve_length = [eval_cumulated_curve_length(xs[ix], metric) for ix in range(nx)]
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
    y1 = x**0 * 1
    y2 = x**0 * 2
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
