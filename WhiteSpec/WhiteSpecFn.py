def spec_white(resid, exog):
    '''
    White's Two-Moment Specification Test

    Parameters
    ----------
    resid : array_like
        OLS residuals
    exog : array_like
        OLS design matrix

    Returns
    -------
    stat : float
        test statistic
    pval : float
        chi-square p-value for test statistic

    Notes
    -----
    Implements the two-moment specification test described by White's
    Theorem 2 (1980, p. 823) which compares the standard OLS covariance
    estimator with White's heteroscedasticity-consistent estimator. The
    test statistic is shown to be chi-square distributed.

    Assumes the OLS design matrix contains an intercept term and at least
    one variable. The intercept is removed to calculate the test statistic.

    Degrees-of-freedom = nvar + nvar * (nvar + 1) / 2

    Reference
    ---------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
    estimator and a direct test for heteroscedasticity. Econometrica,
    48: 817-838.
    '''
    x = np.asarray(exog)
    e = np.asarray(resid)
    if x.ndim == 1:
        raise ValueError('x should have a constant and at least one variable')
    nvar = x.shape[1] - 1
    i0, i1 = np.triu_indices(x.shape[1])
    exog = np.delete(x[:,i0]*x[:,i1], 0, 1)
    assert exog.shape[1] == nvar + nvar * (nvar + 1) / 2
    sqe = e * e
    sqmndevs = sqe - np.mean(sqe)
    D = np.dot(exog.T, sqmndevs)
    devx = exog - np.mean(exog, axis=0)
    B = np.dot(np.dot(devx.T, np.diag(np.square(sqmndevs))), devx)
    stat = np.dot(np.dot(D, np.linalg.inv(B)), D)
    dof = devx.shape[1]
    pval = chisqprob(stat, dof)
    return stat, pval
