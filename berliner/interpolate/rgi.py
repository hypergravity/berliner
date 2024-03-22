"""
A wrapper of ``scipy.interpolate.RegularGridInterpolator``.

References
----------
https://docs.scipy.org/doc/scipy/tutorial/interpolate.html
"""
import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator

__all__ = ["RGI"]


def flat_to_mesh(x_flat: npt.NDArray) -> list[npt.NDArray]:
    # determine ndim
    ndim = x_flat.shape[1]
    # determine grid for each dimension
    grid = [np.unique(x_flat[:, idim]) for idim in range(ndim)]
    # construct mesh (a list of cubes)
    mesh = np.meshgrid(*grid, indexing="ij")
    return mesh


def mesh_to_flat(*xmi: npt.NDArray) -> npt.NDArray:
    # get shape
    shape = xmi[0].shape
    # determine nelement and ndim
    nelement = np.prod(shape)
    ndim = len(xmi)  # ==len(shape)
    # construct flat
    flat = np.zeros((nelement, ndim), dtype=float)
    # fill each dim
    for idim in range(ndim):
        flat[:, idim] = xmi[idim].flatten()
    return flat


def test_mesh_to_flat_to_mesh():
    x = np.linspace(0, 1, 3)
    y = np.linspace(0, 1, 4)
    z = np.linspace(0, 1, 5)
    xmi = np.meshgrid(x, y, z, indexing="ij")

    xmi_rec = flat_to_mesh(mesh_to_flat(*xmi))
    for i in range(3):
        assert np.all(xmi_rec[i] == xmi[i])
        print("Equal: ===\n", xmi_rec[i], xmi[i])


class RGI:
    def __init__(
        self,
        xmi,
        values,
        method="cubic",
        bounds_error=False,
        fill_value=np.nan,
    ):
        self.rgi = RegularGridInterpolator(
            points=xmi,
            values=values,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value,
        )

    def __call__(self, *args, **kwargs):
        return self.rgi(*args, **kwargs)

    def interp_mesh(self, *xmi):
        return self.rgi(mesh_to_flat(*xmi)).reshape(xmi[0].shape)

    def interp_flat(self, flat):
        return self.rgi(flat)

    @staticmethod
    def flat_to_mesh(x_flat: npt.NDArray) -> list[npt.NDArray]:
        return flat_to_mesh(x_flat)

    @staticmethod
    def mesh_to_flat(*xmi: npt.NDArray) -> npt.NDArray:
        return mesh_to_flat(*xmi)
