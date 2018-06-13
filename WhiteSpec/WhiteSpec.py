import numpy as np
from scipy import stats
import pandas as pd
import sys
import os

def prep_white_spec_design_matrix(design, licheck='none'):
    '''
    Prepare the model design matrix for White specification test.              
    
    Parameters
    ----------
    design : array_like
        regression design matrix
    licheck : string ['none', 'cs', 'qr']
        exclude linearly dependent interaction terms
    
    Returns
    -------
    exog : array_like
        modified design matrix as described above

    Notes
    -----
    Structure of the design matrix is exactly the same as used in the White
    heteroscedasticity test, excluding the intercept term. Specifically:
    original IVs, squares of the IVs, and IV cross-product terms.

    User has the option of removing linearly dependent interaction terms,
    e.g., when the model includes dummy variables, etc. Two methods of
    checking linear dependence are provided: QR decomposition and
    Cauchy-Schwartz comparison.

    Reference
    ---------
    http://en.wikipedia.org/wiki/Cauchy-Schwarz_inequality
    https://en.wikipedia.org/wiki/QR_decomposition
    '''
    if design.ndim == 1:
        raise ValueError('X should have a constant and at least one variable')
    nvar1 = design.shape[1] - 1
    idx1, idx2 = np.triu_indices(design.shape[1])
    # delete intercept term
    exog = np.delete(design[:,idx1]*design[:,idx2], 0, 1)
    exogfr = nvar1 + nvar1 * (nvar1 + 1) / 2
    assert exog.shape[1] == exogfr
    atol=1e-14; rtol=1e-13
    if licheck == 'qr':
        print("QR linear independence check")
        r = np.linalg.qr(exog, mode='r')
        tol = atol + rtol * exog.var(0)
        mask = np.abs(r.diagonal()) < np.sqrt(tol)
        exog = exog[:,np.where(~mask)[0]]
    elif licheck == 'cs':
        print("Cauchy-Schwartz linear independence check")
        idx1, idx2 = np.triu_indices(exog.shape[1], 1)
        e1 = exog[:,idx1]; e2 = exog[:,idx2]
        diff = np.abs(np.abs(np.sum(e1*e2, axis=0)) - \
            np.linalg.norm(e1,axis=0)*np.linalg.norm(e2,axis=0))
        mask = diff < atol
        unq = np.unique(mask*idx2)
        exog = exog[:,np.isin(np.arange(exog.shape[1]), unq[unq>0], invert=True)]
    else:
        print("No linear independence check")
    if exog.shape[1] < exogfr:
        print("WARNING: White specification test average covariance matrix is singular.\
               \n         One or more interaction terms has been removed to complete the test.\
               \n         The results of this test should be carefully interpreted.")
    return exog

def spec_white(resid, exog, licheck='none'):
    '''
    White's Two-Moment Specification Test

    Parameters
    ----------
    resid : array_like
        OLS residuals
    exog : array_like
        OLS design matrix
    licheck : string ['none', 'cs', 'qr']
        exclude linearly dependent interaction terms

    Returns
    -------
    dof : int
        degrees of freedom
    stat : float
        test statistic
    pval : float
        chi-square p-value for test statistic

    Notes
    -----
    Implements the two-moment specification test described by White's Theorem
    2 (1980, p. 823) which compares the standard OLS covariance estimator with
    White's heteroscedasticity-consistent estimator. The test statistic is
    shown to be chi-square distributed.

    H0 : the model is homoscedastic and correctly specified

    Assumes the OLS design matrix is full rank and contains an intercept term
    and at least one independent variable.

    Degrees-of-freedom = nvar + nvar * (nvar + 1) / 2

    User has the option of removing linearly dependent interaction terms,
    e.g., when the model includes dummy variables, etc. Two methods of
    checking linear dependence are provided: QR decomposition and
    Cauchy-Schwartz comparison.

    Reference
    ---------
    White, H. (1980). A heteroskedasticity-consistent covariance matrix
    estimator and a direct test for heteroscedasticity. Econometrica, 48:
    817-838.
    '''
    x = prep_white_spec_design_matrix(np.asarray(exog), licheck)
    e = np.asarray(resid)
    sqe = np.square(e)
    sqmndevs = sqe - np.mean(sqe)
    D = np.dot(x.T, sqmndevs)
    devx = x - np.mean(x, axis=0)
    #B = np.linalg.multi_dot([devx.T, np.diag(np.square(sqmndevs)), devx])
    devx *= sqmndevs[:,None]
    B = devx.T.dot(devx)
    #stat = np.linalg.multi_dot([D, np.linalg.inv(B), D])
    stat = D.dot(np.linalg.solve(B, D))
    dof = devx.shape[1]
    pval = stats.chi2.sf(stat, dof)
    return stat, pval, dof

#
# toy program to test White's specification test.
#
# input:
#   .xlsx file containing 4 small test cases.
#   model3 + model4 contain dummy variables.
#
# reference output (SAS 9.3):
#   model1 : dof = 5  stat = 3.25  pval = 0.6613
#   model2 : dof = 9  stat = 6.07  pval = 0.7330
#   model3 : dof = 7  stat = 6.77  pval = 0.4535
#   model4 : dof = 11 stat = 8.46  pval = 0.6714
#
def main():
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    run_dir = os.path.join(cur_dir, "..\\rundir\\")
    files = ['model1.csv', 'model2.csv', 'model3.csv', 'model4.csv']
    for file in files:
        mdl_file = os.path.join(run_dir, file)
        mdl = np.asarray(pd.read_csv(mdl_file))
        # DV is in last column
        lastcol = mdl.shape[1] - 1
        dv = mdl[:, lastcol]
        # create design matrix
        design = np.concatenate((np.ones((mdl.shape[0], 1)), \
            np.delete(mdl, lastcol, 1)), axis=1)
        # perform OLS and generate residuals
        resids = dv - np.dot(design, np.linalg.lstsq(design, dv, rcond=-1)[0])
        # perform White spec test. wspec3/wspec4 contain dummies.
        wres = spec_white(resids, design, 'cs')
        print("White spec(cs): dof =", wres[0], " stat =", wres[1], " pval =", wres[2])
        # compare results to SAS 9.3 output
        if file == 'model1.csv':
            np.testing.assert_almost_equal(wres, [3.251, 0.661, 5], decimal=3)
        elif file == 'model2.csv':
            np.testing.assert_almost_equal(wres, [6.070, 0.733, 9], decimal=3)
        elif file == 'model3.csv':
            np.testing.assert_almost_equal(wres, [6.767, 0.454, 7], decimal=3)
        else:
            np.testing.assert_almost_equal(wres, [8.462, 0.671, 11], decimal=3)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
