
#%%
import numpy as np
from astropy import table
from berliner.utils.machine import general_search_v2 as general_search


DEFAULT_MAGLIM = {
    'umag': (13.0, 24.5),  # sdss
    'gmag': (13.0, 25.0),
    'rmag': (13.0, 24.5),
    'imag': (13.0, 24.0),
    'zmag': (12.0, 22.0),
    'gP1mag': (14.0, 22.0),
    'rP1mag': (14.0, 21.8),
    'iP1mag': (14.5, 21.5),
    'zP1mag': (13.5, 20.9),
    'yP1mag': (12.5, 19.7),
    # 'wP1mag': None,
    'gRmag': None,  # IPHAS, err=0?
    'gImag': None,
    'Hamag': None,
    'Jmag': (9.5, np.inf),
    'Hmag': (9.0, np.inf),
    'Ksmag': (9.0, np.inf),
    # 'IRAC_3.6mag': None,
    # 'IRAC_4.5mag': None,
    # 'IRAC_5.8mag': None,
    # 'IRAC_8.0mag': None,
    # 'MIPS_24mag': None,
    # 'MIPS_70mag': None,
    # 'MIPS_160mag': None,
    'W1mag': (8.0, np.inf),  # 16.5, depth
    'W2mag': (7.0, np.inf),  # 17
    'W3mag': (3.0, np.inf),  # 12.87
    'W4mag': (3.0, np.inf),  # 9.5
    'Gmag': None,               # Gaia, not need to cut
    'G_BPmag': None,
    'G_RPmag': None,
    # 'UXmag': None,
    # 'BXmag': None,
    # 'Bmag': None,
    # 'Vmag': None,
    # 'Rmag': None,
    # 'Imag': None,
    # 'Kmag': None,
    # 'Lmag': None,
    # "L'mag": None,
    # 'Mmag': None,
}


class Pepsi:
    """ PEPSI 
    Photometrically estimate parameters of Stars
    """
    params = None
    sed_mod = None         # stacked isochrone table

    mags = None         # mag names
    maglim = DEFAULT_MAGLIM

    lnprior = None

    Alambda = None

    nband = 0
    ntemp = 0
    ndim = 0

    def __init__(self, params, sed_mod, mags=None, Alambda=None, lnprior=None):
        """ initialize with stacked isochrone table and mag names """
        self.params = np.array(params)
        self.sed_mod = np.array(sed_mod)

        # all available mag systems
        self.nband = self.sed_mod.shape[1]
        self.ntemp = self.sed_mod.shape[0]
        self.ndim = self.params.shape[1]

        self.set_mags(mags)
        self.set_Alambda(Alambda)
        self.set_lnprior(lnprior)

    def set_lnprior(self, lnprior=None):
        """ any kind of prior * dV, in log form
        
        prior, e.g., IMF, galaxy / star counts model
        dV = d(Minit)*d(age)*d([M/H])
        """
        if lnprior is not None:
            self.lnprior = np.array(lnprior)

    def set_Alambda(self, Alambda=None):
        if Alambda is not None:
            self.Alambda = np.array(Alambda)

    def set_mags(self, mags=None):
        if mags is not None:
            self.mags = mags

    def update_maglim(self, new_maglim):
        """ update maglim """
        self.maglim.update(new_maglim)

    def preproc(self, sed_obs):
        sed_obs = np.array(sed_obs)

        if sed_obs.ndim == 1:
            # single sed
            for i in range(self.nband):
                this_mag = self.mags[i]
                this_maglim = self.maglim[this_mag]
                if this_maglim is not None:
                    if not this_maglim[0]<=sed_obs[i]<=this_maglim[1]:
                        sed_obs[i] = np.nan
        else:
            # multiple sed
            for i in range(self.nband):
                this_mag = self.mags[i]
                this_maglim = self.maglim[this_mag]
                if this_maglim is not None:
                    sed_obs[:,i] = np.where((sed_obs[:,i]>=this_maglim[0])&(sed_obs[:,i]<=this_maglim[1]), sed_obs[:,i], np.nan)

        return sed_obs

    def evaluate(self, sed_obs, sed_obs_err, vpi_obs=None, vpi_obs_err=None,
                 Lvpi=1.0, Lprior=1.0,
                 cost_order=2, av_llim=-0.001, debug=False, verbose=False):

        sed_obs = self.preproc(sed_obs)

        if sed_obs.ndim == 1:

            r = general_search(
                self.params, self.sed_mod, self.lnprior, self.Alambda,
                sed_obs, sed_obs_err=sed_obs_err,
                vpi_obs=vpi_obs, vpi_obs_err=vpi_obs_err,
                Lvpi=Lvpi, Lprior=Lprior,
                cost_order=cost_order, av_llim=av_llim, debug=debug)

            return r

        else:
            rs = []
            for i in range(sed_obs.shape[0]):
                if verbose:
                    print(i)
                r = general_search(
                    self.params, self.sed_mod, self.lnprior, self.Alambda,
                    sed_obs[i], sed_obs_err=sed_obs_err[i],
                    vpi_obs=vpi_obs[i], vpi_obs_err=vpi_obs_err[i],
                    Lvpi=Lvpi, Lprior=Lprior,
                    cost_order=cost_order, av_llim=av_llim, debug=debug)
                rs.append(r)

            return rs

    def __repr__(self):
        return "< PEPSI >"

