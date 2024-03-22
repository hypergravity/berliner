"""
Translated from Aaron Dotter: https://github.com/aarondotter/iso/blob/master/src/eep.f90
"""
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import Iterable, Optional
from astropy.table import Table
from collections import OrderedDict

COLNAMES = {
    "star_age": "star_age",
    "star_mass": "star_mass",
    "log_LH": "log_LH",
    "log_LHe": "log_LHe",
    "log_Teff": "log_Teff",
    "log_L": "log_L",
    "log_g": "log_g",
    "log_center_T": "log_center_T",
    "log_center_Rho": "log_center_Rho",
    "center_h1": "center_h1",
    "center_he4": "center_he4",
    "center_c12": "center_c12",
    "center_gamma": "center_gamma",
    "surface_h1": "surface_h1",
    "he_core_mass": "he_core_mass",
    "c_core_mass": "c_core_mass",
}


class EEPNotFoundError(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)


class PrimaryEEP:
    def __init__(self):
        self.EEPs = {}  # so far not used

    @staticmethod
    def get_EEPs_low_mass(track: Table):
        idx_PreMS = PrimaryEEP.get_PreMS_EEP(track)
        idx_ZAMS = PrimaryEEP.get_ZAMS_EEP(track, idx_PreMS=idx_PreMS)
        idx_IAMS = PrimaryEEP.get_IAMS_EEP(track, idx_ZAMS=idx_ZAMS)
        idx_TAMS = PrimaryEEP.get_TAMS_EEP(track, idx_IAMS=idx_IAMS)
        idx_RGBTip = PrimaryEEP.get_RGBTip_EEP(track, idx_TAMS=idx_TAMS)
        idx_ZAHB = PrimaryEEP.get_ZAHB_EEP(track, idx_RGBTip=idx_RGBTip)
        idx_TAHB = PrimaryEEP.get_TAHB_EEP(track, idx_ZAHB=idx_ZAHB)
        idx_TPAGB = PrimaryEEP.get_TPAGB_EEP(track, idx_TAHB=idx_TAHB)
        idx_PostAGB = PrimaryEEP.get_PostAGB_EEP(track, idx_TPAGB=idx_TPAGB)
        idx_WDCS = PrimaryEEP.get_WDCS_EEP(track, idx_PostAGB=idx_PostAGB)
        return dict(
            idx_PreMS=idx_PreMS,
            idx_ZAMS=idx_ZAMS,
            idx_IAMS=idx_IAMS,
            idx_TAMS=idx_TAMS,
            idx_RGBTip=idx_RGBTip,
            idx_ZAHB=idx_ZAHB,
            idx_TAHB=idx_TAHB,
            idx_TPAGB=idx_TPAGB,
            idx_PostAGB=idx_PostAGB,
            idx_WDCS=idx_WDCS,
        )

    @staticmethod
    def get_EEPs_high_mass(track: Table):
        idx_PreMS = PrimaryEEP.get_PreMS_EEP(track)
        idx_ZAMS = PrimaryEEP.get_ZAMS_EEP(track, idx_PreMS=idx_PreMS)
        idx_IAMS = PrimaryEEP.get_IAMS_EEP(track, idx_ZAMS=idx_ZAMS)
        idx_TAMS = PrimaryEEP.get_TAMS_EEP(track, idx_IAMS=idx_IAMS)
        idx_RGBTip = PrimaryEEP.get_RGBTip_EEP(track, idx_TAMS=idx_TAMS)
        idx_ZAHB = PrimaryEEP.get_ZAHB_EEP(track, idx_RGBTip=idx_RGBTip)
        idx_TAHB = PrimaryEEP.get_TAHB_EEP(track, idx_ZAHB=idx_ZAHB)
        idx_Cburn = PrimaryEEP.get_CBurn_EEP(track, idx_TAHB=idx_TAHB)
        return dict(
            idx_PreMS=idx_PreMS,
            idx_ZAMS=idx_ZAMS,
            idx_IAMS=idx_IAMS,
            idx_TAMS=idx_TAMS,
            idx_RGBTip=idx_RGBTip,
            idx_ZAHB=idx_ZAHB,
            idx_TAHB=idx_TAHB,
            idx_Cburn=idx_Cburn,
        )

    @staticmethod
    def get_PreMS_EEP(track: Table) -> int:
        """the first point exceeding log_center_T=5"""
        log_center_T = 5.0

        for i in range(len(track)):
            if track[i][COLNAMES["log_center_T"]] > log_center_T:
                return i
        return -1

    @staticmethod
    def get_ZAMS_EEP(track: Table, idx_PreMS: int = 0, kind: int = 3) -> int:
        """LH/L>0.999 while Xc>Xc_init*0.9985"""
        if idx_PreMS < 0:
            return -1

        if kind == 1:
            # Dotter+2016 paper version: 0.0015 of initial H are burnt
            LH_fraction = 0.999
            center_h1_fraction = 0.9985
            center_h1_threshold = track[0][COLNAMES["center_h1"]] * center_h1_fraction

            for i in range(idx_PreMS, len(track)):
                if (
                    track[i][COLNAMES["log_LH"]] - track[i][COLNAMES["log_L"]]
                    > np.log10(LH_fraction)
                    and track[i][COLNAMES["center_h1"]] > center_h1_threshold
                ):
                    return i
        elif kind == 3:
            # iso/src/eep.f90 version: argmax of log_g before 0.001 of initial H are burnt
            Xmax = track[idx_PreMS][COLNAMES["center_h1"]]
            Xmin = Xmax - 1e-3
            ZAMS1 = (
                np.where(track[idx_PreMS:][COLNAMES["center_h1"]] < Xmin)[0][0]
                + idx_PreMS
            )
            ZAMS3 = np.argmax(track[idx_PreMS:ZAMS1][COLNAMES["log_g"]]) + idx_PreMS
            return ZAMS3

        return -1

    @staticmethod
    def get_IAMS_EEP(track: Table, idx_ZAMS: int = 0) -> int:
        """Xc=0.3"""
        if idx_ZAMS < 0:
            return -1

        # the first point below Xc=0.3
        Xc = 0.3

        for i in range(idx_ZAMS, len(track)):
            if track[i][COLNAMES["center_h1"]] < Xc:
                return i

        # for very-low-mass stars, (<0.5Msun)
        if (
            track[0][COLNAMES["star_mass"]] < 0.5
            and track[-1][COLNAMES["star_age"]] > 1.5e10
        ):
            return len(track) - 1
        return -1

    @staticmethod
    def get_TAMS_EEP(track: Table, idx_IAMS: int = 0) -> int:
        """Xc=1e-12"""
        if idx_IAMS < 0:
            return -1

        # the first point below Xc=1e-12
        Xc = 1e-12

        for i in range(idx_IAMS, len(track)):
            if track[i][COLNAMES["center_h1"]] < Xc:
                return i

        # for very-low-mass stars, (<0.5Msun)
        if (
            track[0][COLNAMES["star_mass"]] < 0.5
            and track[-1][COLNAMES["star_age"]] > 1.5e10
        ):
            return len(track) - 1
        return -1

    @staticmethod
    def get_RGBTip_EEP(track: Table, idx_TAMS: int = 874) -> int:
        """minTeff or maxL while Yc>Tc_TAMS-0.01"""
        if idx_TAMS < 0:
            return -1

        center_he4_tams = track[idx_TAMS][COLNAMES["center_he4"]]

        # Yc drop 0.01 by fraction
        idx_center_he4_drop = -1
        for idx_center_he4_drop in range(idx_TAMS, len(track)):
            if (
                track[idx_center_he4_drop][COLNAMES["center_he4"]]
                < center_he4_tams - 0.01
            ):
                break
        if idx_center_he4_drop < 0:
            return -1

        # min Teff before Yc drop
        idx_min_Teff = np.argmin(
            track[COLNAMES["log_Teff"]][idx_TAMS:idx_center_he4_drop]
        )
        # max L before Yc drop
        idx_max_L = np.argmax(track[COLNAMES["log_L"]][idx_TAMS:idx_center_he4_drop])
        # which comes first
        return idx_TAMS + min(idx_min_Teff, idx_max_L)

    @staticmethod
    def get_ZAHB_EEP(track: Table, idx_RGBTip: int = 14815) -> int:
        """minTc while Yc>Tc_RGBTip-0.03"""
        if idx_RGBTip < 0:
            return -1

        center_he4_RGBTip = track[idx_RGBTip][COLNAMES["center_he4"]]

        idx_center_he4_drop = -1
        for idx_center_he4_drop in range(idx_RGBTip, len(track)):
            if (
                track[idx_center_he4_drop][COLNAMES["center_he4"]]
                < center_he4_RGBTip - 0.03
            ):
                break

        if idx_center_he4_drop < 0:
            return -1
        idx_min_center_T = np.argmin(
            track[COLNAMES["log_center_T"]][idx_RGBTip:idx_center_he4_drop]
        )
        return idx_RGBTip + idx_min_center_T

    @staticmethod
    def get_TAHB_EEP(track: Table, idx_ZAHB: int = 14815) -> int:
        """central He mass falls below 1e-4 Msun"""
        if idx_ZAHB < 0:
            return -1

        for i in range(idx_ZAHB, len(track)):
            if track[i][COLNAMES["center_he4"]] < 1e-4:
                return i
        return -1

    @staticmethod
    def get_CBurn_EEP(track: Table, idx_TAHB: int = 14815) -> int:
        """central C mass fraction falls below 1e−4"""
        # for high-mass stars only
        for i in range(idx_TAHB, len(track)):
            if (
                track[i][COLNAMES["center_h1"]] < 1e-8
                and track[i][COLNAMES["center_he4"]] < 1e-8
                and track[i][COLNAMES["center_c12"]] < 1e-4
            ):
                return i
        return -1

    @staticmethod
    def get_TPAGB_EEP(track: Table, idx_TAHB: int = 14815) -> int:
        """He shell mass < 0.1Msun"""
        if idx_TAHB < 0:
            return -1
        for i in range(idx_TAHB, len(track)):
            if (
                track[i][COLNAMES["center_he4"]] < 1e-6  # Yc<10^-6
                and track[i][COLNAMES["he_core_mass"]]  # He shell mass < 0.1 Msun
                - track[i][COLNAMES["c_core_mass"]]
                < 0.1
            ):
                return i
        return -1

    @staticmethod
    def get_PostAGB_EEP(track: Table, idx_TPAGB: int = 14815) -> int:
        """CO core mass > 80% stellar mass"""
        if idx_TPAGB < 0:
            return -1
        Tc_TPAGB = track[idx_TPAGB][COLNAMES["log_center_T"]]
        Tc_end = track[-1][COLNAMES["log_center_T"]]
        if Tc_TPAGB > Tc_end:
            # has post-AGB
            for i in range(idx_TPAGB, len(track)):
                if (
                    track[i][COLNAMES["c_core_mass"]] / track[i][COLNAMES["star_mass"]]
                    > 0.8
                ):
                    return i
            return -1
        # has no post-AGB
        return -1

    @staticmethod
    def get_WDCS_EEP(track: Table, idx_PostAGB: int = 14815) -> int:
        """central gamma > 100"""
        if idx_PostAGB < 0:
            return -1
        # center_gamma
        center_gamma_limit = 100
        for i in range(idx_PostAGB, len(track)):
            if track[i][COLNAMES["center_gamma"]] > center_gamma_limit:
                return i
        return -1


def test():
    t = Table.read("exdata/r7507-mist-tracks/m1.00.fits")
    eeps = PrimaryEEP.get_EEPs_low_mass(t)

    print(eeps)
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    plt.plot(np.log10(np.diff(t["star_age"])))
    # plt.plot(t_v12s["log_dt"])
    plt.plot(t["log_Teff"], label="log_Teff")
    # plt.plot(t_v12s["log_L"], label="log_L")
    # plt.plot(t_v12s["center_h1"], label="h1")
    plt.plot(t["center_he4"], label="h4")
    plt.plot(t["log_L"], label="log_L")
    plt.plot(t["log_LH"], label="log_LH")
    plt.plot(t["log_LH"] - t["log_L"], label="log_LH-log_L")
    plt.plot(t["log_center_T"], label="log_center_T")
    plt.plot(t["he_core_mass"], label="he_core_mass", lw=3)
    plt.plot(t["c_core_mass"], label="c_core_mass", lw=3)
    ax.vlines([eeps[k] for k in eeps.keys()], -4, 4, color="k", lw=5)
    for k in eeps.keys():
        ax.text(eeps[k], 5, k[4:])
        ax.text(eeps[k], 6, t["star_age"][eeps[k]], rotation=90)
    plt.legend()
    fig.tight_layout()
