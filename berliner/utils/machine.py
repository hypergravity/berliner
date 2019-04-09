import numpy as np
from emcee import EnsembleSampler


class TGMMachine():
    """ the TGM Machine class """

    def __init__(self, isoc_stacked, tgm_cols=("teff", "logg", "_mhini"),
                 tgm_sigma=(100, 0.2, 0.1), w_sigma=(100, 0.2, 0.1),
                 pred_cols=("teff", "logg"), wcol="w"):
        """ the input should be a stacked table of isochrones """

        self.data = isoc_stacked  # stacked isochrones
        self.tgm = np.array(isoc_stacked[tgm_cols].to_pandas())  # TGM array
        self.tgm_sigma = np.array(tgm_sigma).reshape(1, -1)  # TGM sigma
        self.pred_array = np.array(
            self.data[pred_cols].to_pandas())  # Qs to predict
        self.w = isoc_stacked[wcol].data  # weight array
        self.w_sigma = np.array(w_sigma).reshape(1, -1)

    def predict(self, test_tgm):
        """ predict MLE of SED and weight at the given TGM position """
        test_tgm = np.array(test_tgm).reshape(1, -1)
        test_w = self.w * np.exp(-0.5 * np.sum(((self.tgm - test_tgm) / self.tgm_sigma) ** 2., axis=1))
        pred_result = np.sum(self.pred_array * test_w.reshape(-1, 1), axis=0) / np.sum(test_w)
        # smooth weight in a wider volume
        test_w = self.w * np.exp(-0.5 * np.sum(((self.tgm - test_tgm) / self.w_sigma) ** 2., axis=1))
        return pred_result, np.sum(test_w)


class SED2TG():
    """ the TG Machine class """

    def __init__(self, r, Alambda, p_bounds=None, phot_bands=[]):
        self.r = r
        self.Alambda = Alambda
        self.phot_bands = phot_bands
        self.p_bounds = p_bounds

    def predict(self, *args, **kwargs):
        sampler = self.runsample(*args, **kwargs)
        return np.median(sampler.flatchain, axis=0), np.std(sampler.flatchain, axis=0)

    def runsample(self, sed_obs, sed_obs_err, vpi_obs, vpi_obs_err,
                  Lvpi=1.0, Lprior=1.0, nsteps=(1000, 1000, 2000), p0try=None):

        ndim = 4                # 4 stands for [Teff, logg, Av, DM]
        nwalkers = len(p0try)   # number of chains

        for i in range(len(nsteps)):
            if i == 0:
                # initialize sampler
                sampler = EnsembleSampler(nwalkers, ndim, costfun,
                                          args=(self.r, self.p_bounds,
                                                self.Alambda, sed_obs,
                                                sed_obs_err, vpi_obs,
                                                vpi_obs_err, Lvpi, Lprior))
                # guess Av and DM for p0try
                p0try = np.array([initial_guess(_, self.r, self.Alambda, sed_obs, sed_obs_err) for _ in p0try])
                # run sampler
                pos, _, __ = sampler.run_mcmc(p0try, nsteps[i])
            else:
                # generate new p
                p_rand = random_p(sampler, nloopmax=1000, method="mle",
                                  costfun=costfun, args=(self.r, self.p_bounds,
                                                         self.Alambda, sed_obs,
                                                         sed_obs_err, vpi_obs,
                                                         vpi_obs_err,
                                                         Lvpi, Lprior))
                # reset sampler
                sampler.reset()
                # run at new p
                pos1, lnprob1, rstate1 = sampler.run_mcmc(p_rand, nsteps[i])
        return sampler


def model_sed_abs(x, r):
    """ interpolate with r(Regli) at position x """
    test_tgm = x[:2]
    return r(test_tgm)[:-1]


def initial_guess(x, r, Alambda, sed_obs, sed_obs_err):
    """ initial guess of Av and DM with OLS method """
    # select good bands
    ind_good_band = np.isfinite(sed_obs) & (sed_obs_err > 0)
    sed_mod = model_sed_abs(x, r).reshape(1, -1)[:, ind_good_band]
    sed_obs = sed_obs.reshape(1, -1)[:, ind_good_band]

    # solve Av and DM
    av_est, dm_est = guess_avdm(sed_mod, sed_obs, Alambda[ind_good_band])

    # neg Av --> 0.001
    if av_est <= 0:
        av_est = 0.001

    return np.array([x[0], x[1], av_est, dm_est])


def costfun(x, r, p_bounds, Alambda, sed_obs, sed_obs_err, vpi_obs, vpi_obs_err, Lvpi, Lprior):
    """ cost function of MCMC

    Returns
    -------
    -0.5*(chi2_sed + chi2_vpi*Lvpi) + lnprior*Lprior

    """
    ind_good_band = np.isfinite(sed_obs) & (sed_obs_err > 0)

    # unpack parameters
    test_tg = x[:2]
    test_av = x[2]
    test_dm = x[3]

    # check bounds
    if p_bounds is not None:
        if not check_bounds(x, p_bounds):
            return -np.inf

    # predict model
    pred_mod = r(test_tg)

    # lnprior
    lnprior = pred_mod[-1]

    # predicted SED_obs
    sed_mod = pred_mod[:-1] + Alambda * test_av + test_dm

    # vpi_model
    vpi_mod = 10 ** (2 - 0.2 * test_dm)  # mas

    # chi2_sed
    chi2_sed = np.nansum(
        (((sed_obs - sed_mod) / sed_obs_err) ** 2.)[ind_good_band])
    if not np.isfinite(chi2_sed):
        return -np.inf

    # include vpi
    if Lvpi > 0:
        # eval chi2_vpi
        chi2_vpi = ((vpi_obs - vpi_mod) / vpi_obs_err) ** 2.

        if np.isfinite(chi2_vpi):
            return -0.5 * (chi2_sed + chi2_vpi) + lnprior*Lprior
        else:
            return -0.5 * chi2_sed + lnprior*Lprior
    else:
        return -0.5 * chi2_sed + lnprior*Lprior


def mcostfun(*args):
    """ minus of costfun """
    return -costfun(*args)


def generate_p(p0, pstd, shrink=0.5):
    """ generate (normal) random p """
    return p0 + shrink * pstd * np.random.randn(len(p0))


def check_bounds(p, p_bounds=None):
    """ check bounds """
    if p_bounds is not None:
        p_bounds = np.array(p_bounds)
        if np.any(np.array(p) <= p_bounds[:, 0]) or np.any(np.array(p) >= p_bounds[:, 1]):
            return False
    return True


def random_p(sampler, nloopmax=1000, method="mle", costfun=None, args=()):
    """ given a sampler, generate new random p """
    n_walkers, _, n_dim = sampler.chain.shape

    # MLE p
    if method == "mle":
        p_mle = sampler.flatchain[np.nanargmax(sampler.flatlnprobability)]
    else:
        p_mle = np.median(sampler.flatchain, axis=0)
    # STD p
    p_std = np.std(sampler.flatchain, axis=0)

    # generate new p
    p_rand = []
    for i in range(nloopmax):
        p_new = generate_p(p_mle, p_std, shrink=0.6)
        if not np.isfinite(costfun(p_new, *args)):
            continue
        else:
            p_rand.append(p_new)
        if i == nloopmax - 1:
            raise (ValueError("Unable to get good random ps..."))
        if len(p_rand) >= n_walkers:
            break
    if len(p_rand) == n_walkers:
        return np.array(p_rand)
    else:
        raise (ValueError("random_p failed!"))


def guess_avdm(sed_mod, sed_obs, Alambda):
    """ guess Av and DM with OLS method
    Parameters
    ----------
    sed_mod:
        (n_band, ) array
    sed_obs:
        (n_band, ) array
    """

    n_band = sed_obs.size
    # X = [[Alambda_i, 1], [], ...]
    X = np.matrix(np.ones((n_band, 2), float))
    X[:, 0] = Alambda[:, None]
    # Y = [[d_sed_i], [], ...]
    Y = np.matrix((sed_obs - sed_mod).reshape(-1, 1))

    # OLS solution
    av_ols, dm_ols = np.array(np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y))
    #av_est, dm_est = np.array(np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y))

    return np.array([av_ols, dm_ols])
