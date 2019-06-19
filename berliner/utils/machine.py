import numpy as np
from emcee import EnsembleSampler

from .ls import wls_simple


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

    def grid_search(self, test_sed_obs, test_sed_obs_err=None,
                    test_vpi_obs=None, test_vpi_obs_err=None,
                    Lvpi=1.0, Lprior=1.0, sed_err_typical=0.1,
                    cost_order=2, av_llim=0., return_est=False):
        p_mle, p_mean, p_std = grid_search2(self.r, self.Alambda,
                                            test_sed_obs, test_sed_obs_err,
                                            test_vpi_obs, test_vpi_obs_err,
                                            Lvpi, Lprior, sed_err_typical,
                                            cost_order, av_llim, return_est)
        sed_mean = self.r(p_mean[:2])[:-1]+self.Alambda*p_mean[2]+p_mean[3]
        sed_rmse = np.sqrt(np.nanmean(np.square(sed_mean-test_sed_obs)))
        return p_mle, p_mean, p_std, sed_rmse


def grid_search2(r2, Alambda,
                 test_sed_obs, test_sed_obs_err=None,
                 test_vpi_obs=None, test_vpi_obs_err=None,
                 Lvpi=1.0, Lprior=1.0, sed_err_typical=0.1, cost_order=2,
                 av_llim=0., return_est=False):
    """
    when p = [T, G, Av, DM],
    given a set of SED,
    find the best T, G and estimate the corresponding Av and DM
    """

    # select good bands
    if test_sed_obs_err is None:
        # all bands will be used
        ind_good_band = np.isfinite(test_sed_obs)
    else:
        ind_good_band = np.isfinite(test_sed_obs) & (test_sed_obs_err > 0)

    n_good_band = np.sum(ind_good_band)
    if n_good_band < 5:
        return [np.ones((4,),)*np.nan for i in range(3)]

    # lnprior
    lnprior = r2.values[:, -1]

    # T & G grid
    t_est, g_est = r2.flats.T

    # model SED
    sed_mod = r2.values[:, :-1][:, ind_good_band]
    # observed SED
    sed_obs = test_sed_obs[ind_good_band]
    # observed SED error
    if sed_err_typical is not None:
        sed_obs_err = np.ones_like(sed_obs, float)*sed_err_typical
    else:
        sed_obs_err = test_sed_obs_err[ind_good_band]

    # WLS to guess Av and DM
    av_est, dm_est = guess_avdm_wls(
        sed_mod, sed_obs, sed_obs_err, Alambda[ind_good_band])

    # cost(SED)
    res_sed = sed_mod + av_est.reshape(-1, 1) * Alambda[ind_good_band] + dm_est.reshape(-1, 1) - sed_obs
    if sed_err_typical is not None:
        cost_sed = np.nansum(np.abs(res_sed / sed_err_typical) ** cost_order, axis=1)
    else:
        cost_sed = np.nansum(np.abs(res_sed / sed_obs_err) ** cost_order, axis=1)
    lnprob = -0.5 * cost_sed

    # cost(VPI)
    if test_vpi_obs is not None and test_vpi_obs_err is not None and Lvpi > 0:
        vpi_mod = 10 ** (2 - 0.2 * dm_est)
        cost_vpi = ((vpi_mod - test_vpi_obs) / test_vpi_obs_err) ** 2.
        if np.all(np.isfinite(cost_vpi)):
            lnprob -= 0.5*cost_vpi

    # lnprob = cost(SED) + cost(VPI) + prior
    if Lprior > 0:
        lnprob += lnprior * Lprior

    # eliminate neg Av
    lnprob[av_est < av_llim] = -np.inf
    lnprob -= np.nanmax(lnprob)

    if return_est:
        return t_est, g_est, av_est, dm_est, cost_sed, lnprob

    # normalization
    prob = np.exp(lnprob)
    prob /= np.sum(prob)

    # weighted mean
    av_mle = av_est[np.argmax(lnprob)]
    dm_mle = dm_est[np.argmax(lnprob)]
    t_mle = t_est[np.argmax(lnprob)]
    g_mle = g_est[np.argmax(lnprob)]

    av_mean = np.sum(av_est * prob)
    dm_mean = np.sum(dm_est * prob)
    t_mean = np.sum(t_est * prob)
    g_mean = np.sum(g_est * prob)

    av_std = np.sum((av_est - av_mean) ** 2 * prob)
    dm_std = np.sum((dm_est - dm_mean) ** 2 * prob)
    t_std = np.sum((t_est - t_mean) ** 2 * prob)
    g_std = np.sum((g_est - g_mean) ** 2 * prob)

    p_mle = np.array([t_mle, g_mle, av_mle, dm_mle])
    p_mean = np.array([t_mean, g_mean, av_mean, dm_mean])
    p_std = np.array([t_std, g_std, av_std, dm_std])

    return p_mle, p_mean, p_std


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
    av_est, dm_est = guess_avdm_ols(sed_mod, sed_obs, Alambda[ind_good_band])

    # neg Av --> 0.001
    if av_est <= 0:
        av_est = 0.001

    return np.array([x[0], x[1], av_est, dm_est])


def guess_avdm_ols(sed_mod, sed_obs, Alambda):
    """ matrix form OLS solution for Av and DM """
    sed_mod = np.array(sed_mod)
    sed_obs = np.array(sed_obs)

    assert sed_mod.ndim == 2
    assert sed_obs.ndim == 2

    n_band = sed_obs.size
    # color
    X = np.array([Alambda, np.ones_like(Alambda)]).T
    y = np.matrix((sed_obs - sed_mod).T)
    # av_ols, dm_ols = np.array(np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y))
    av_est, dm_est = np.array(
        np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y))

    return av_est, dm_est


def guess_avdm_wls(sed_mod, sed_obs, sed_obs_err, Alambda):
    """ matrix form OLS solution for Av and DM """
    sed_mod = np.array(sed_mod)
    sed_obs = np.array(sed_obs)
    sed_obs_err = np.array(sed_obs_err)

    assert sed_mod.ndim == 2

    # d_mag
    X = np.array([Alambda, np.ones_like(Alambda, float)]).T
    y = (sed_obs.reshape(1, -1) - sed_mod).T
    yerr = sed_obs_err

    # solve Av & DM with WLS
    av_est, dm_est = wls_simple(X, y, yerr)

    return av_est, dm_est


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


# def guess_avdm(sed_mod, sed_obs, Alambda):
#     """ guess Av and DM with OLS method
#     Parameters
#     ----------
#     sed_mod:
#         (n_band, ) array
#     sed_obs:
#         (n_band, ) array
#     """
#
#     n_band = sed_obs.size
#     # X = [[Alambda_i, 1], [], ...]
#     X = np.matrix(np.ones((n_band, 2), float))
#     X[:, 0] = Alambda[:, None]
#     # Y = [[d_sed_i], [], ...]
#     Y = np.matrix((sed_obs - sed_mod).reshape(-1, 1))
#
#     # OLS solution
#     av_ols, dm_ols = np.array(np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y))
#     #av_est, dm_est = np.array(np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y))
#
#     return np.array([av_ols, dm_ols])

def general_search(params, sed_mod, lnprior,
                   Alambda,
                   test_sed_obs, test_sed_obs_err=None,
                   test_vpi_obs=None, test_vpi_obs_err=None,
                   Lvpi=1.0, Lprior=1.0, sed_err_typical=0.1, cost_order=2,
                   av_llim=0., debug=False):
    """
    when p = [T, G, Av, DM],
    given a set of SED,
    find the best T, G and estimate the corresponding Av and DM
    """

    # select good bands
    if test_sed_obs_err is None:
        # all bands will be used
        ind_good_band = np.isfinite(test_sed_obs)
    else:
        ind_good_band = np.isfinite(test_sed_obs) & (test_sed_obs_err > 0)

    n_good_band = np.sum(ind_good_band)
    if n_good_band < 5:
        return [np.ones((4,), ) * np.nan for i in range(3)]

    # lnprior
    # lnprior = r2.values[:, -1]

    # T & G grid
    # t_est, g_est = r2.flats.T
    # params

    # model SED
    # sed_mod = r2.values[:, :-1][:, ind_good_band]
    sed_mod = sed_mod[:, ind_good_band]
    # observed SED
    sed_obs = test_sed_obs[ind_good_band]
    # observed SED error
    if sed_err_typical is not None:
        sed_obs_err = np.ones_like(sed_obs, float) * sed_err_typical
    else:
        sed_obs_err = test_sed_obs_err[ind_good_band]

    # WLS to guess Av and DM
    av_est, dm_est = guess_avdm_wls(
        sed_mod, sed_obs, sed_obs_err, Alambda[ind_good_band])

    # cost(SED)
    res_sed = sed_mod + av_est.reshape(-1, 1) * Alambda[
        ind_good_band] + dm_est.reshape(-1, 1) - sed_obs

    if sed_err_typical is not None:
        cost_sed = np.nansum(np.abs(res_sed / sed_err_typical) ** cost_order,
                             axis=1)
    else:
        cost_sed = np.nansum(np.abs(res_sed / sed_obs_err) ** cost_order,
                             axis=1)
    lnprob = -0.5 * cost_sed

    # cost(VPI)
    if test_vpi_obs is not None and test_vpi_obs_err is not None and Lvpi > 0:
        vpi_mod = 10 ** (2 - 0.2 * dm_est)
        cost_vpi = ((vpi_mod - test_vpi_obs) / test_vpi_obs_err) ** 2.
        if np.all(np.isfinite(cost_vpi)):
            lnprob -= 0.5 * cost_vpi

    # lnprob = cost(SED) + cost(VPI) + prior
    if Lprior > 0:
        lnprob += lnprior * Lprior

    # eliminate neg Av
    lnprob[av_est < av_llim] = -np.inf
    lnprob -= np.nanmax(lnprob)

    if debug:
        return params, av_est, dm_est, cost_sed, lnprob

    # normalization
    prob = np.exp(lnprob)
    prob /= np.sum(prob)

    # weighted mean
    ind_mle = np.argmax(lnprob)
    av_mle = av_est[ind_mle]
    dm_mle = dm_est[ind_mle]
    p_mle = params[ind_mle]

    av_mean = np.sum(av_est * prob)
    dm_mean = np.sum(dm_est * prob)
    p_mean = np.sum(params * prob.reshape(-1, 1), axis=0)

    av_std = np.sum((av_est - av_mean) ** 2 * prob)
    dm_std = np.sum((dm_est - dm_mean) ** 2 * prob)
    p_std = np.sum((params - p_mean) ** 2 * prob.reshape(-1, 1), axis=0)

    p_mle = np.hstack([p_mle, av_mle, dm_mle])
    p_mean = np.hstack([p_mean, av_mean, dm_mean])
    p_std = np.hstack([p_std, av_std, dm_std])

    rms_sed_mle = np.sqrt(np.nanmean(res_sed[ind_mle] ** 2.))
    rms_sed_min = np.min(np.sqrt(np.nanmean(res_sed ** 2., axis=1)))

    return dict(
        p_mle=p_mle,
        p_mean=p_mean,
        p_std=p_std,
        rmsmle=rms_sed_mle,
        rmsmin=rms_sed_min,
        ind_mle=ind_mle,
        n_good=np.sum(ind_good_band)
    )


def general_search_v2(params, sed_mod, lnprior, Alambda,
                      sed_obs, sed_obs_err=0.1,
                      vpi_obs=None, vpi_obs_err=None,
                      Lvpi=1.0, Lprior=1.0,
                      cost_order=2, av_llim=-0.001, debug=False):
    """
    when p = [teff, logg, [M/H], Av, DM], theta = [teff, logg, [M/H]],
    given a set of SED,
    find the best theta and estimate the corresponding Av and DM
    """

    n_band = len(sed_obs)
    n_mod = sed_mod.shape[0]

    # cope with scalar sed_obs_err
    if isinstance(sed_obs_err, np.float):
        sed_obs_err = np.ones_like(sed_obs, np.float) * sed_obs_err

    # select good bands
    ind_good_band = np.isfinite(sed_obs) & (sed_obs_err > 0)
    n_good_band = np.sum(ind_good_band)
    if n_good_band < 4:
        # n_good_band = 3: unique solution
        # so n_good_band should be at least 4
        return [np.ones((4,), ) * np.nan for i in range(3)]

    # use a subset of bands
    sed_mod_select = sed_mod[:, ind_good_band]
    # observed SED
    sed_obs_select = sed_obs[ind_good_band]
    sed_obs_err_select = sed_obs_err[ind_good_band]
    # extinction coefs
    Alambda_select = Alambda[ind_good_band]

    # WLS to guess Av and DM
    av_est, dm_est = guess_avdm_wls(
        sed_mod_select, sed_obs_select, sed_obs_err_select, Alambda_select)

    # cost(SED)
    res_sed = sed_mod_select + av_est.reshape(-1, 1) * Alambda_select \
        + dm_est.reshape(-1, 1) - sed_obs_select
    lnprob_sed = -0.5 * np.nansum(
        np.abs(res_sed / sed_obs_err_select) ** cost_order, axis=1)

    # cost(VPI)
    if vpi_obs is not None and vpi_obs_err is not None and Lvpi > 0:
        vpi_mod = 10 ** (2 - 0.2 * dm_est)
        lnprob_vpi = -0.5 * ((vpi_mod - vpi_obs) / vpi_obs_err) ** 2.
    else:
        lnprob_vpi = np.zeros((n_mod,), np.float)
    lnprob_vpi = np.where(np.isfinite(lnprob_vpi), lnprob_vpi, 0) * Lvpi

    # lnprob = cost(SED) + cost(VPI) + prior
    if Lprior > 0:
        lnprob_prior = lnprior * Lprior

    # posterior probability
    lnpost = lnprob_sed + lnprob_vpi + lnprob_prior
    # eliminate neg Av
    lnpost[av_est < av_llim] = -np.inf
    lnpost -= np.nanmax(lnpost)

    # for debugging the code
    if debug:
        return dict(params=params,
                    av_est=av_est,
                    dm_est=dm_est,
                    lnprob_sed=lnprob_sed,
                    lnprob_vpi=lnprob_vpi,
                    lnprior=lnprior)

    # normalization
    post = np.exp(lnpost)
    L0 = np.sum(post)

    # weighted mean
    # ind_mle = np.argmax(lnpost)
    # av_mle = av_est[ind_mle]
    # dm_mle = dm_est[ind_mle]
    # p_mle = params[ind_mle]

    L1_av = np.sum(av_est * post)
    L1_dm = np.sum(dm_est * post)
    L1_p = np.sum(params * post.reshape(-1, 1), axis=0)

    L2_av = np.sum(av_est ** 2 * post)
    L2_dm = np.sum(dm_est ** 2 * post)
    L2_p = np.sum(params ** 2 * post.reshape(-1, 1), axis=0)

    sigma_av = np.sqrt(L2_av / L0 - L1_av ** 2 / L0 ** 2)
    sigma_dm = np.sqrt(L2_dm / L0 - L1_dm ** 2 / L0 ** 2)
    sigma_p = np.sqrt(L2_p / L0 - L1_p ** 2 / L0 ** 2)

    # MLE model
    ind_mle = np.argmax(lnprob_sed + lnprob_vpi)
    av_mle = av_est[ind_mle]
    dm_mle = dm_est[ind_mle]
    p_mle = params[ind_mle]

    p_mle = np.hstack([p_mle, av_mle, dm_mle])
    p_mean = np.hstack([L1_p/L0, L1_av/L0, L1_dm/L0])
    p_err = np.hstack([sigma_p, sigma_av, sigma_dm])

    rms_sed_mle = np.sqrt(np.nanmean(res_sed[ind_mle] ** 2.))
    rms_sed_min = np.min(np.sqrt(np.nanmean(res_sed ** 2., axis=1)))

    return dict(p_mle=p_mle,
                p_mean=p_mean,
                p_err=p_err,
                rmsmle=rms_sed_mle,
                rmsmin=rms_sed_min,
                ind_mle=ind_mle,
                n_good=np.sum(ind_good_band))
