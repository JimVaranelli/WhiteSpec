# WhiteSpec
Python implementation of White's specification test. Version contained in WhiteSpec/WhiteSpecFn.py has been submitted to statsmodels package.

## Parameters
resid : array_like
    OLS residuals
exog : array_like
    OLS design matrix

## Returns
stat : float
    test statistic
pval : float
    chi-square p-value for test statistic
dof : int
    degrees of freedom

## Notes
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

## Reference
White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroscedasticity. Econometrica, 48: 817-838.

## Requirements
Python 3.6
Numpy 1.13.1
Scipy 0.19.1
Pandas 0.20.3

## Running
There are no parameters. The program is set up to access test files in the ..\rundir directory. This path can be modified in the source file.

## Additional Info
Please see comments in the source file for additional info including referenced output for the test file.
