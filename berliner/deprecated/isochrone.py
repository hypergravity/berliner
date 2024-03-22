import numpy as np


class Isochrone:
    def mock_photometry(self):
        pass

    def interpolate(self):
        pass

    @staticmethod
    def assign_mag_error(mag, relation="csst"):
        return


class CMD2:
    def __init__(self, mag):
        pass

    @staticmethod
    def from_mag(self, mag):
        return CMD2()

    def cmd_likelihood(self, cmd) -> float:
        return 0


def test():
    # sample MassFunction -> mass
    mass = IMF().sample()
    # interpolate isochrone -> mag
    mag_with_error = Isochrone().mock_photometry()
    # convert photometry to CMD
    cmd_mod = CMD2.from_mag(mag_with_error)
    cmd_obs = CMD2.from_mag(mag_obs)
    # compare two CMDs
    return cmd_mod.likelihood(cmd_obs)
