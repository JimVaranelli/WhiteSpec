# WhiteSpec
Python implementation of White's specification test

## Reference
White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroscedasticity. Econometrica, 48: 817-838.

## Description
A Python program that implements the two-moment specification test described by White's theorem 2 (1980, p. 823) which compares the standard OLS covariance estimator with White's heteroscedasticity-consistent estimator. The comparison statistic is shown to be chi-square distributed. Degrees-of-freedom is the number of regressors, not including the intercept.

H0 : model is homoscedastic and correctly specified

The user has the option of removing linearly dependent interaction terms which typically arise when the model includes dummy variables.

## Requirements
Python 3.6
Numpy 1.13.1
Scipy 0.19.1
Pandas 0.20.3

## Running
There are no parameters. The program is set up to access a test file in the ..\rundir directory. This path can be modified in the source file.

## Additional Info
Please see comments in the source file for additional info including referenced output for the test file.
