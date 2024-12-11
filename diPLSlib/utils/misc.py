'''
Some helper functions for diPLSlib
'''

import numpy as np
from scipy.stats import norm 
from scipy.stats import f
from math import exp, sqrt
from scipy.special import erf

def gengaus(length, mu, sigma, mag, noise=0):
    """
    Generate a Gaussian spectrum-like signal with optional random noise.

    Parameters
    ----------

    length : int
        Length of the generated signal.

    mu : float
        Mean of the Gaussian function.

    sigma : float
        Standard deviation of the Gaussian function.

    mag : float
        Magnitude of the Gaussian signal.

    noise : float, optional (default=0)
        Standard deviation of the Gaussian noise to be added to the signal.

    Returns
    -------

    signal : ndarray of shape (length,)
        The generated Gaussian signal with noise.

    Examples
    --------

    >>> from diPLSlib.utils.misc import gengaus
    >>> import numpy as np
    >>> import scipy.stats
    >>> signal = gengaus(100, 50, 10, 5, noise=0.1)
    """

    s = mag*norm.pdf(np.arange(length),mu,sigma)
    n = noise*np.random.rand(length)
    signal = s + n

    return signal


def hellipse(X, alpha=0.05): 
    """
    Compute the 95% confidence interval ellipse for a 2D scatter plot.

    Parameters
    ----------

    X : ndarray of shape (n_samples, 2)
        Matrix of data points.

    alpha : float, optional (default=0.05)
        Significance level for the confidence interval.

    Returns
    -------

    el : ndarray of shape (2, 100)
        Coordinates of the ellipse's points. To plot, use `plt.plot(el[0, :], el[1, :])`.

    Examples
    --------

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from diPLSlib.utils.misc import hellipse
    >>> X = np.random.random((100, 2))
    >>> el = hellipse(X)
    >>> plt.scatter(X[:,0], X[:,1], label='Data points')            # doctest: +ELLIPSIS
    <matplotlib.collections.PathCollection object at ...>
    >>> plt.plot(el[0,:], el[1,:], label='95% Confidence Ellipse')  # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.legend()                                                # doctest: +ELLIPSIS
    <matplotlib.legend.Legend object at ...>
    """
    
    # Means
    mean_all = np.zeros((2,1))   
    mean_all[0] = np.mean(X[:,0])
    mean_all[1] = np.mean(X[:,1])

    # Covariance matrix
    X = X[:,:2]
    comat_all = np.cov(np.transpose(X))

    # SVD
    U,S,V = np.linalg.svd(comat_all)

    # Confidence limit computed as the 95% quantile of the F-Distribution
    N = np.shape(X)[0]
    quant = 1 - alpha
    Conf = (2*(N-2))/(N-2)*f.ppf(quant,2,(N-2))
    
    # Evalute CI on (0,2pi)
    el = np.zeros((2,100))
    t = np.linspace(0,2*np.pi,100)
    for j in np.arange(100):
        sT = np.matmul(U,np.diag(np.sqrt(S*Conf)))
        el[:,j] = np.transpose(mean_all)+np.matmul(sT,np.array([np.cos(t[j]),np.sin(t[j])]))   

    return el


def rmse(y, yhat):
    """
    Compute the Root Mean Squared Error (RMSE) between two arrays.

    Parameters
    ----------

    y : ndarray of shape (n_samples,)
        True values.

    yhat : ndarray of shape (n_samples,)
        Predicted values.

    Returns
    -------

    error : ndarray of shape (n_samples,)
        The RMSE between `y` and `yhat`.

    Examples
    --------

    >>> import numpy as np
    >>> from diPLSlib.utils.misc import rmse
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([2, 3, 4])
    >>> error = rmse(x, y)
    >>> print(error)
    1.0
    """

    return np.sqrt(((y.ravel()-yhat.ravel())**2).mean())


def calibrateAnalyticGaussianMechanism(epsilon, delta, GS, tol = 1.e-12):
    """ 
    Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]

    Parameters
    ----------
    epsilon : float
        Privacy parameter epsilon

    delta : float
        Desired privacy failure probability

    GS : float
        Upper bound on the L2-sensitivity of the function to which the mechanism is applied

    tol : float
        Error tolerance for binary search


    Returns
    -------
    sigma : float
        Standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity GS

    References
    ----------
    - Balle, B., & Wang, Y. X. (2018, July). Improving the gaussian mechanism for differential privacy: Analytical calibration and optimal denoising. In International Conference on Machine Learning (pp. 394-403). PMLR.

    Examples
    --------
    >>> from diPLSlib.utils.misc import calibrateAnalyticGaussianMechanism
    >>> calibrateAnalyticGaussianMechanism(1.0, 1e-5, 1.0)
    3.730631634944469
    """

    def Phi(t):
        return 0.5*(1.0 + erf(float(t)/sqrt(2.0)))

    def caseA(epsilon,s):
        return Phi(sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def caseB(epsilon,s):
        return Phi(-sqrt(epsilon*s)) - exp(epsilon)*Phi(-sqrt(epsilon*(s+2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while(not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0*s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup-s_inf)/2.0
        while(not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup-s_inf)/2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s : caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s : caseA(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) - sqrt(s/2.0)

        else:
            predicate_stop_DT = lambda s : caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s : caseB(epsilon, s)
            predicate_left_BS = lambda s : function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s : sqrt(1.0 + s/2.0) + sqrt(s/2.0)

        predicate_stop_BS = lambda s : abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)
        
    sigma = alpha*GS/sqrt(2.0*epsilon)

    return sigma