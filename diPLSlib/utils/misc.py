'''
Some helper functions for diPLSlib
'''

import numpy as np
from scipy.stats import norm 
from scipy.stats import f

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
    >>> print(signal)
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
    >>> plt.scatter(X[:,0], X[:,1], label='Data points')
    >>> plt.plot(el[0,:], el[1,:], label='95% Confidence Ellipse')
    >>> plt.legend()
    >>> plt.show()
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
    """

    return np.sqrt(((y-yhat)**2).mean())