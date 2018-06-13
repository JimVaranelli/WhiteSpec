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
    dof : int
        degrees of freedom

    Notes
    -----
    Implements the two-moment specification test described by White's
    Theorem 2 (1980, p. 823) which compares the standard OLS covariance
    estimator with White's heteroscedasticity-consistent estimator. The
    test statistic is shown to be chi-square distributed.

    Null hypothesis is homoscedastic and correctly specified.

    Assumes the OLS design matrix contains an intercept term and at least
    one variable. The intercept is removed to calculate the test statistic.

    Interaction terms (squares and crosses of OLS regressors) are added to
    the design matrix to calculate the test statistic.

    Degrees-of-freedom (full rank) = nvar + nvar * (nvar + 1) / 2

    Linearly dependent columns are removed to avoid singular matrix error.

    Reference
    ---------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
    estimator and a direct test for heteroscedasticity. Econometrica,
    48: 817-838.
    '''
    x = np.asarray(exog)
    e = np.asarray(resid)
    if x.ndim == 1:
        raise ValueError('X should have a constant and at least one variable')
    nvar = x.shape[1] - 1

    # add interaction terms
    i0, i1 = np.triu_indices(x.shape[1])
    exog = np.delete(x[:,i0] * x[:,i1], 0, 1)
    assert exog.shape[1] == nvar + nvar * (nvar + 1) / 2

    # collinearity check - see _fit_collinear
    atol=1e-14; rtol=1e-13
    tol = atol + rtol * exog.var(0)
    r = np.linalg.qr(exog, mode='r')
    mask = np.abs(r.diagonal()) < np.sqrt(tol)
    exog = exog[:,np.where(~mask)[0]]

    # calculate test statistic
    sqe = e * e
    sqmndevs = sqe - np.mean(sqe)
    D = np.dot(exog.T, sqmndevs)
    devx = exog - np.mean(exog, axis=0)
    devx *= sqmndevs[:, None]
    B = devx.T.dot(devx)
    stat = D.dot(np.linalg.solve(B, D))

    # chi-square test
    dof = devx.shape[1]
    pval = stats.chi2.sf(stat, dof)
    return stat, pval, dof
