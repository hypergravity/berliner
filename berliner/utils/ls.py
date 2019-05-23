
import numpy as np


def ols(X, y):
    """
    ordinary least squares: X*beta = y
    solution: beta = (X.T*X)^-1 * X.T *y

    Note
    ----
    ols solves multiple problems (n_problems>1 allowed)


    Parameters
    ----------
    X:
        (n_obs, n_term) array
    y:
        (n_obs, n_problems) array
    Return
    ------
    beta:
        (n_term, n_problems) array

    """

    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
    return beta


def wls(X, W, y):
    """
    weighted least squares: (X.T*W*X)*beta = X.T*W*y
    solution: beta = (X.T*X)^-1 * X.T *y

    Note
    ----
    wls solves single problems (n_problems=1)
    BUT! is able to solve multiple-template (same error) problems

    Parameters
    ----------
    X: predictors
        (n_obs, n_term) array
    W: weight matrix / inverse covariance matrix
        (n_obs, n_obs) weight matrix
    y: response
        (n_obs, n_problems) array

    Return
    ------
    beta: coefficients
        (n_term, 1) array

    """

    if W.ndim == 1:
        np.diagflat(W)

    W = np.where(W > 0, W, 1e-10)

    Xp = np.matmul(np.matmul(X.T, W), X)
    yp = np.matmul(np.matmul(X.T, W), y)
    beta = ols(Xp, yp)

    return beta


def wls_simple(X, y, yerr):
    """
    weighted least squares: (X.T*W*X)*beta = X.T*W*y
    solution: beta = (X.T*X)^-1 * X.T *y

    Note
    ----
    wls solves single problems (n_problems=1)
    BUT! is able to solve multiple-template (same error) problems

    Parameters
    ----------
    X: predictors
        (n_obs, n_term) array
    yerr: error of response
        (n_obs, ) weight matrix
    y: response
        (n_obs, n_problems) array

    Return
    ------
    beta: coefficients
        (n_term, ) array

    """

    yerr = yerr.reshape(-1, 1)  # column vector
    yerr = np.where((yerr > 0) & np.isfinite(yerr), yerr, 1e5)

    X_ = X / yerr
    y_ = y / yerr
    beta = ols(np.matmul(X_.T, X_), np.matmul(X_.T, y_))

    return beta
