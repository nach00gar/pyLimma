import pandas as pd
import numpy as np
from numpy.linalg import det, inv
from scipy.stats import t
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.api as sm
from scipy.special import psi, polygamma
import itertools

def DEGsExtraction(mat, labels, lfc, pvalue, coverage=1, contrasts=None):
    """
    Extracts differentially expressed genes (DEGs) from a given matrix.

    Parameters:
    - mat (pd.DataFrame): Gene Expression matrix with genes as rows and samples as columns.
    - labels (np.array): Array of sample labels.
    - lfc (float): Log fold change threshold.
    - pvalue (float): p-value threshold for significance.
    - coverage (int, optional): Coverage parameter, default is 1.
    - contrasts (np.array, optional): Contrast matrix, default is None.

    Returns:
    - data : DataFrame containing the results of DEGs analysis (LFC and p-values).
    - ind : Array with positions from the selected DEGs.
    """

    if len(np.unique(labels)) == 2:
        X = np.vstack((np.ones(labels.shape).squeeze(), pd.get_dummies(labels, dtype="int").iloc[:, 1])).T
    else:
        X = pd.get_dummies(labels, dtype="int")
        X.columns = np.unique(labels)
    
        if contrasts is None:
            contrasts = makeContrasts(len(X.columns))
        else:
            if contrasts.shape[0] != len(X.columns):
                raise Exception("Incorrect size of contrasts")

    c=contrasts
    
    stack = []
    sigmas = []
    dfs = []

    for i in range(len(mat)):
        y = mat.iloc[i].to_numpy()
        results = sm.regression.linear_model.OLS(y, X).fit()
        stack.append(results.params)
        sigmas.append(np.sqrt(results.mse_resid))
        dfs.append(results.df_resid)
        
    params = np.array(stack)
    if len(np.unique(labels)) == 2:
        stdev_unscaled = np.diag(np.sqrt(inv(X.T @ X)))
    else:
        stdev_unscaled = np.diag(np.sqrt(c.T @ inv(X.T @ X) @ c))
        
    aux = np.ones((params.shape[0], stdev_unscaled.shape[0]))
    aux *= stdev_unscaled
    
    fit = dict()
    fit['coefficients'] = params
    fit['stdev_unscaled'] = aux
    fit['sigma'] = np.array(sigmas)
    fit['df_residual'] = np.array(dfs)
    
    if len(np.unique(labels)) > 2:
        fit['coefficients'] = fit['coefficients'] @ c
    print(fit['coefficients'].shape)    
    fit['coefficients'] = np.asarray(fit['coefficients'])
    
    d = empirical_bayes_statistics(fit)
    
    if len(np.unique(labels)) > 2:
        vals = np.array([fdrcorrection(d['p_value'][:, i], alpha=pvalue, method='indep', is_sorted=False)[1] for i in range(d['p_value'].shape[1])]).T
        res = np.zeros(shape=fit["coefficients"].shape)

        res[vals <= pvalue] = 1
        neg = fit["coefficients"] < 0
        res[neg] = res[neg] * -1

        th = abs(fit["coefficients"]) < lfc
        res[th] = 0

        non_zero_counts = np.sum(res != 0, axis=1)
        ind = np.where(non_zero_counts > coverage)[0]

        data = pd.DataFrame(dtype='float64')
        for i in range(fit["coefficients"].shape[1]):
            data[f'lFC{i+1}'] = pd.Series(fit["coefficients"][:, i])

        for i, p_adj in enumerate(vals.T):
            data[f'adj_p{i+1}'] = pd.Series(p_adj)

        data = data.assign(Gene=mat.index).set_index("Gene")
    
    else:
        vals = fdrcorrection(d['p_value'][:, 1], alpha=pvalue, method='indep', is_sorted=False)[1]
        data = pd.DataFrame(dtype='float64')
        data['lFC'] = pd.Series(fit["coefficients"][:, 1])
        data['adj_p'] = pd.Series(vals)
        data = data.assign(Gene=mat.index).set_index("Gene")
        
        ind = np.where((abs(data['lFC']) >= lfc) & (data['adj_p'] <= pvalue))[0]      
    
    
    
    return data, ind


def empirical_bayes_statistics(fit):
    """
    Performs empirical Bayes statistics on the fitted model.

    Parameters:
    - fit (dict): Dictionary containing the fitted model parameters.

    Returns:
    - out: Dictionary containing empirical Bayes statistics and p-values.
    """

    coefficients = fit['coefficients']
    stdev_unscaled = fit['stdev_unscaled']
    sigma = fit['sigma']
    df_residual = fit['df_residual']
    
    if coefficients is None or stdev_unscaled is None or sigma is None or df_residual is None:
        raise ValueError("No data, or any element of previous fit is None")
    
    if np.max(df_residual) == 0:
        raise ValueError("No residual degrees of freedom in linear model fits")
    
    if not np.any(np.isfinite(sigma)):
        raise ValueError("No finite residual standard deviations")
    
    # Moderated t-statistic
    s2 = np.power(sigma, 2)
    out = squeeze_var(s2, df_residual)
    out['s2_prior'] = out['var_prior']
    out['s2_post'] = out['var_post']
    out.pop('var_prior', None)
    out.pop('var_post', None)
    
    t_statistic = np.zeros(coefficients.shape)
    mid = coefficients / stdev_unscaled
    t_statistic = mid / np.sqrt(out['s2_post'])[:, np.newaxis]
    df_total = df_residual + out['df_prior']
    df_pooled = np.sum(df_residual, axis=0)
    df_total = np.minimum(df_total, df_pooled)

    p_value = 2 * t.cdf(-np.abs(t_statistic), df=df_total[0]) 
    
    out['t'] = t_statistic
    out['df_total'] = df_total
    out['p_value'] = p_value
    
    return out

def squeeze_var(var, df):
    n = len(var)

    # Casos especiales degenerados
    if n == 0:
        raise ValueError("var is empty")
    if n == 1:
        return {'var_post': var, 'var_prior': var, 'df_prior': 0}

    if len(df) > 1:
        var[df == 0] = 0

    fit = fit_f_dist(var, df1=df)
    df_prior = fit['df2']
    if np.any(np.isnan(df_prior)):
        raise ValueError("Could not estimate prior")

    var_post = squeeze_var_helper(var=var, df=df, var_prior=fit['scale'], df_prior=df_prior)

    return {'df_prior': df_prior, 'var_prior': fit['scale'], 'var_post': var_post}

def squeeze_var_helper(var, df, var_prior, df_prior):
    n = len(var)
    is_finite = np.isfinite(df_prior)

    if np.all(is_finite):
        return (df * var + df_prior * var_prior) / (df + df_prior)

    if len(var_prior) == n:
        var_post = var_prior
    else:
        var_post = np.repeat(var_prior, n)

    if np.any(is_finite):
        indices = np.where(is_finite)[0]
        if len(df) > 1:
            df = df[indices]
        df_prior = df_prior[indices]
        var_post[indices] = (df * var[indices] + df_prior * var_post[indices]) / (df + df_prior)

    return var_post



def trigammaInverse(x):
    """
    Inverts the trigamma function via Newton Raphson procedure.

    Parameters:
    - x (np.array): Array of data values.

    Returns:
    - np.array: Inverted trigamma values.
    """
    omit = np.isnan(x)
    if np.any(omit):
        y = x.copy()
        if np.any(~omit):
            y[~omit] = trigammaInverse(x[~omit])
        return y

    omit = (x < 0)
    if np.any(omit):
        y = x.copy()
        y[omit] = np.nan
        print("NaNs produced")
        if np.any(~omit):
            y[~omit] = trigammaInverse(x[~omit])
        return y

    omit = (x > 1e7)
    if np.any(omit):
        y = x.copy()
        y[omit] = 1 / np.sqrt(x[omit])
        if np.any(~omit):
            y[~omit] = trigammaInverse(x[~omit])
        return y

    omit = (x < 1e-6)
    if np.any(omit):
        y = x.copy()
        y[omit] = 1 / x[omit]
        if np.any(~omit):
            y[~omit] = trigammaInverse(x[~omit])
        return y

    y = 0.5 + 1 / x
    iter = 0
    while True:
        iter += 1
        tri = polygamma(1, y)
        dif = tri * (1 - tri / x) / polygamma(2, y)
        y += dif
        if np.max(-dif / y) < 1e-8:
            break
        if iter > 50:
            print("Iteration limit exceeded")
            break
    return y


def fit_f_dist(x, df1):
    """
    Fits an F-distribution to the given data. Internal step of eBayes procedure.

    Parameters:
    - x (np.array): Array of data values.
    - df1 (np.array): Degrees of freedom for the F-distribution.

    Returns:
    - dict: Dictionary containing the scale and df2 parameters of the F-distribution.
    """
    o = np.isfinite(x) & np.isfinite(df1) & (x >= 0) & (df1 > 0)
    if np.any(~o):
        x = x[o]
        df1 = df1[o]
    n = len(x)
    if n == 0:
        return {'scale': np.nan, 'df2': np.nan}

    m = np.median(x)
    if m == 0:
        print("More than half of the residual vars are zero: eBayes process is less reliable")
        m = 1
    elif np.any(x == 0):
        print("Sample variances equal to zero were detected, they have been shifted")
    x = np.maximum(x, 1e-5 * np.median(x))

    z = np.log(x)
    e = z - psi(df1/2) + np.log(df1/2)
    emean = np.mean(e)
    evar = np.mean(n / (n - 1) * (e - emean) ** 2 - polygamma(1, df1/2))
    if evar > 0:
        df2 = 2 * trigammaInverse(evar)
        s20 = np.exp(emean + psi(df2/2) - np.log(df2/2))
    else:
        df2 = np.inf
        s20 = np.exp(emean)
    
    return {'scale': s20, 'df2': df2}



def makeContrasts(n_classes):
    """
    Creates a contrast matrix for the given number of classes, by default output all (n_classes^2 - n_classes)/2 contrasts.

    Parameters:
    - n_classes (int): Number of classes.

    Returns:
    - np.array: Contrast matrix.
    """
    combinations = list(itertools.combinations(range(n_classes), 2))
    n_combinations = len(combinations)
    contrast_matrix = np.zeros((n_combinations, n_classes))
    
    for idx, (i, j) in enumerate(combinations):
        contrast_matrix[idx, i] = 1
        contrast_matrix[idx, j] = -1
        
    return contrast_matrix.T