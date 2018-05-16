import numpy as np
from scipy.stats import chisqprob
from pandas import ExcelFile
import sys

#
# prepare the model design matrix for White specification test.
# structure of the design matrix is exactly the same as used in the
# White heteroscedasticity test, excluding the intercept term.
# specifically: original IVs, squares of the IVs, and IV cross-product
# terms.
#
# user has the option of removing linearly dependent interaction terms
# which typically arise when the model includes dummy variables.
#
# parameters:
#   design : array_like
#     regression design matrix
#   dummy : boolean
#     exclude linearly dependent interaction terms
#
# returns:
#   design : array_like
#     modified design matrix as described above
#   
def prep_white_spec_design_matrix(design, dummy):
    if dummy == False:
      idx1, idx2 = np.triu_indices(design.shape[1])
      design = np.delete(design[:,idx1]*design[:,idx2], 0, 1)
      return design
    # delete the intercept terms from design matrix
    design = np.delete(design, 0, 1)
    cols = design.shape[1]
    lindep = np.zeros(cols)
    # find linear dependecies in square/interaction terms
    warn = False
    for i in range (cols):
        col = design[:, i]
        if np.allclose(col, col*col, rtol=0, atol=1e-10):
            lindep[i] = 1
            warn = True
    if warn == True:
        print("WARNING: White specification test average covariance matrix is singular. \
               \n         One or more interaction terms has been removed to complete the test. \
               \n         The results of this test should be carefully interpreted.")
    # add the square and interaction terms to the design matrix
    for i in range (cols):
        # check for square-term linear dependence
        if lindep[i] == 1:
            continue
        col = design[:, i]
        # add the square term
        design = np.append(design, np.reshape(col*col, (np.size(col), 1)), 1)
        # add the interaction terms
        for j in range (i+1, cols):
            # check for interaction-term linear dependence
            if lindep[j] == 1:
                continue
            design = np.append(design, np.reshape(col*design[:, j], (np.size(col), 1)), 1)
    return design

#
# White's specification test.
#
# White, H. (1980). A heteroskedasticity-consistent covariance matrix
# estimator and a direct test for heteroscedasticity. Econometrica,
# 48: 817-838.
#
# implements the two-moment specification test described by White's
# theorem 2 (1980, p. 823) which compares the standard OLS covariance
# estimator with White's heteroscedasticity-consistent estimator. the
# comparison statistic is shown to be chi-square distributed. degrees
# of freedom is the number of regressors, not including the intercept.
#
# H0 : model is homoscedastic and correctly specified
#
# user has the option of removing linearly dependent interaction terms
# which typically arise when the model includes dummy variables.
#
# parameters:
#   design : array_like
#     OLS design matrix
#   resids : array_like
#     OLS residuals
#   dummy : boolean
#     exclude 
#
# returns:
#   stat : float
#     test statistic
#   pval : float
#     chi-square p-value for test statistic
# 
def spec_white(design, resids, dummy=False):
    # prepare design matrix for test
    design = prep_white_spec_design_matrix(design, dummy)
    # calculate the squared-residual mean deviations
    sqresids = resids * resids
    sqmeandevs = sqresids - np.mean(sqresids)
    # calculate mean-difference vector D
    D = np.dot(design.T, sqmeandevs)
    # calculate average covariance matrix B
    devdesign = design - np.mean(design, axis=0)
    B = np.dot(np.dot(devdesign.T, np.diag(np.square(sqmeandevs))), devdesign)
    # calculate the test statistic
    stat = np.dot(np.dot(D, np.linalg.inv(B)), D)
    # degrees of freedom is the number of regressors not including intercept
    dof = devdesign.shape[1]
    pval = chisqprob(stat, dof)
    print("White spec: dof =", dof, " stat =", stat, " pval =", pval)
    return stat, pval

#
# toy program to test White's specification test.
#
# input:
#   .xlsx file containing 4 small test cases.
#   model3 + model4 contain dummy variables.
#
# reference output (SAS 9.3):
#   model1 : df = 5  stat = 3.25  pval = 0.6613
#   model2 : df = 9  stat = 6.07  pval = 0.7330
#   model3 : df = 7  stat = 6.77  pval = 0.4535
#   model4 : df = 11 stat = 8.46  pval = 0.6714
#
def main():
    xlsx = ExcelFile('../rundir/whitespectest.xlsx')
    for sheet in xlsx.sheet_names:
        print("Sheet =", sheet)
        design = np.asarray(xlsx.parse(sheet))
        # DV is in last column
        lastcol = design.shape[1] - 1
        dv = design[:, lastcol]
        # create design matrix
        design = np.concatenate((np.ones((design.shape[0], 1)), np.delete(design, lastcol, 1)), axis=1)
        # perform OLS and generate residuals
        resids = dv - np.dot(design, np.linalg.lstsq(design, dv)[0])
        # perform White spec test. model3/model4 contain dummies.
        if sheet == 'model3' or sheet == 'model4':
            wstat, wpval = spec_white(design, resids, True)
        else:
            wstat, wpval = spec_white(design, resids, False)

if __name__ == "__main__":
    sys.exit(int(main() or 0))