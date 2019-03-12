import numpy as np
from regli.regli import bisect_interval
from regli import Regli


class TrackSet:
    def __init__(self, track_logm, track_feh, tracks):
        """tracks should be a list of astropy.table.Table instances """
        # self.tracks = tracks
        self.grid_logm = np.unique(track_logm)
        self.grid_feh = np.unique(track_feh)
        self.r = Regli(self.grid_logm, self.grid_feh)
        self.ind_dict = self.r.ind_dict

        # sort tracks
        print("@TrackSet: sorting tracks ...")
        tracks_sorted = []
        i_tracks = []
        logm_sorted = []
        feh_sorted = []
        for _logm, _feh in self.r.flats:
            d_track = np.abs(track_logm - _logm) + np.abs(track_feh - _feh)
            i_track = np.argmin(d_track)
            # print(i_track, d_track[i_track])
            i_tracks.append(i_track)
            tracks_sorted.append(tracks[i_track])
            logm_sorted.append(track_logm[i_track])
            feh_sorted.append(track_feh[i_track])

        self.logm = np.array(logm_sorted)
        self.feh = np.array(feh_sorted)
        self.tracks = None
        self.set_tracks(np.array(tracks_sorted))

    def set_tracks(self, tracks):
        self.tracks = tracks

    def interp_tgm(self, x=(1.01, 0.01, 213.89), interp_colnames=["log_Teff", "log_g", "_feh"]):
        data = self.interp(x=x, interp_colnames=interp_colnames)
        data[0] = 10 ** data[0]
        return data

    def interps(self, xs, interp_colnames=["log_Teff", "log_g", "_feh"]):
        """ interplate tracks at multiple points

        Parameters
        ----------
        xs:
            initial mass, [Fe/H], EEP

        """
        return np.array([self.interp(_x, interp_colnames=interp_colnames) for _x in xs])

    def interp(self, x=(1.01, 0.01, 213.89), interp_colnames=["log_Teff", "log_g", "_feh"]):
        """ interplate tracks at a single point

        Parameters
        ----------
        x:
            initial mass, [Fe/H], EEP

        """
        test_mini, test_feh, test_eep = x

        # convert to log scale
        test_logm = np.log10(test_mini)

        # edges
        e_logm = bisect_interval(edges=self.grid_logm, x=test_logm)
        e_feh = bisect_interval(edges=self.grid_feh, x=test_feh)

        # if out of bounds, return nan
        if e_logm[0] < -1 or e_feh[0] < -1:
            return np.nan * np.zeros((len(interp_colnames, )), float)

        # otherwise, find neighbouring tracks
        itrack00 = self.ind_dict[e_logm[0], e_feh[0]]
        itrack01 = self.ind_dict[e_logm[0], e_feh[1]]
        itrack10 = self.ind_dict[e_logm[1], e_feh[0]]
        itrack11 = self.ind_dict[e_logm[1], e_feh[1]]

        # calculate nodes
        p_logm_0 = self.grid_logm[e_logm[0]]
        p_logm_1 = self.grid_logm[e_logm[1]]
        p_feh_0 = self.grid_feh[e_feh[0]]
        p_feh_1 = self.grid_feh[e_feh[1]]

        # determine weights
        v_tot = (p_logm_1 - p_logm_0) * (p_feh_1 - p_feh_0)
        v_00 = (p_logm_1 - test_logm) * (p_feh_1 - test_feh)
        v_01 = (p_logm_1 - test_logm) * (test_feh - p_feh_0)
        v_10 = (test_logm - p_logm_0) * (p_feh_1 - test_feh)
        v_11 = (test_logm - p_logm_0) * (test_feh - p_feh_0)
        # v_00+v_01+v_10+v_11
        w = np.array([v_00, v_01, v_10, v_11]) / v_tot

        these_tracks = self.tracks[[itrack00, itrack01, itrack10, itrack11]]
        eep_min = np.max([t["_eep"][0] for t in these_tracks])
        eep_max = np.min([t["_eep"][-1] for t in these_tracks])

        if test_eep < eep_min or test_eep > eep_max:
            return np.nan * np.zeros((len(interp_colnames, )), float)

        data = np.zeros((len(interp_colnames, )), float)

        for itrack in range(len(these_tracks)):
            ieep = bisect_interval(these_tracks[itrack]["_eep"], x=test_eep)
            w_intrack = np.array(
                [these_tracks[itrack]["_eep"][ieep[1]] - test_eep, test_eep - these_tracks[itrack]["_eep"][ieep[0]]])
            data_intrack = np.zeros((len(interp_colnames, )), float)
            for icol, interp_colname in enumerate(interp_colnames):
                data_intrack[icol] = these_tracks[itrack][interp_colname][ieep[0]] * w_intrack[0] + \
                                     these_tracks[itrack][interp_colname][ieep[1]] * w_intrack[1]
            data += data_intrack * w[itrack]
            # print(data_intrack)
            # print(data)

        return data

    def get_a_track(self, logm, feh):
        """ return the closest track """
        d_track = np.abs(self.logm - logm) + np.abs(self.feh - feh)
        i_track = np.argmin(d_track)
        return self.tracks[i_track]
