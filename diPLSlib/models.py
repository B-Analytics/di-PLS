# -*- coding: utf-8 -*-
'''
diPLSlib model classes

- DIPLS base class
- GCTPLS class
'''

# Modules
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import matplotlib.pyplot as plt
from diPLSlib import functions as algo
from diPLSlib.utils import misc as helpers
import scipy.stats


class DIPLS(RegressorMixin, BaseEstimator):
    """
    Domain-Invariant Partial Least Squares (DIPLS) algorithm for domain adaptation.

    This class implements the DIPLS algorithm, which is designed to align feature distributions 
    across different domains while predicting the target variable `y`. It supports multiple 
    source and target domains through domain-specific feature transformations.

    Parameters
    ----------

    A : int
        Number of latent variables to be used in the model.

    l : Union[int, List[int]], default=0
        Regularization parameter. Either a single value or a list of different
        values for each latent variable (LV).

    centering : bool, default=True
            If True, source and target domain data are mean-centered.

    heuristic : bool, default=False
        If True, the regularization parameter is set to a heuristic value that
        balances fitting the output variable y and minimizing domain discrepancy.

    target_domain : int, default=0
        If multiple target domains are passed, target_domain specifies
        for which of the target domains the model should apply. 
        If target_domain=0, the model applies to the source domain,
        if target_domain=1, it applies to the first target domain, and so on.

    rescale : Union[str, ndarray], default='Target'
            Determines rescaling of the test data. If 'Target' or 'Source', the test data will be
            rescaled to the mean of xt or xs, respectively. If an ndarray is provided, the test data
            will be rescaled to the mean of the provided array.

    Attributes
    ----------

    n : int
        Number of samples in `x`.

    ns : int
        Number of samples in `xs`.

    nt : int
        Number of samples in `xt`.

    k : int
        Number of features (variables) in `x`.

    mu_ : ndarray of shape (n_features,)
        Mean of columns in `x`.

    mu_s_ : ndarray of shape (n_features,)
        Mean of columns in `xs`.

    mu_t_ : ndarray of shape (n_features,) or ndarray of shape (n_domains, n_features)
        Mean of columns in `xt`, averaged per target domain if multiple domains exist.

    b_ : ndarray of shape (n_features, 1)
        Regression coefficient vector.

    b0_ : float
        Intercept of the regression model.

    T_ : ndarray of shape (n_samples, A)
        Training data projections (scores).

    Ts_ : ndarray of shape (n_source_samples, A)
        Source domain projections (scores).

    Tt_ : ndarray of shape (n_target_samples, A)
        Target domain projections (scores).

    W_ : ndarray of shape (n_features, A)
        Weight matrix.

    P_ : ndarray of shape (n_features, A)
        Loadings matrix corresponding to x.

    Ps_ : ndarray of shape (n_features, A)
        Loadings matrix corresponding to xs.

    Pt_ : ndarray of shape (n_features, A)
        Loadings matrix corresponding to xt.

    E_ : ndarray of shape (n_source_samples, n_features)
        Residuals of source domain data.

    Es_ : ndarray of shape (n_source_samples, n_features)
        Source domain residual matrix.

    Et_ : ndarray of shape (n_target_samples, n_features)
        Target domain residual matrix.

    Ey_ : ndarray of shape (n_source_samples, 1)
        Residuals of response variable in the source domain.

    C_ : ndarray of shape (A, 1)
        Regression vector relating source projections to the response variable.

    opt_l_ : ndarray of shape (A, 1)
        Heuristically determined regularization parameter for each latent variable.

    discrepancy_ : ndarray
        The variance discrepancy between source and target domain projections.


    References
    ----------

    1. Ramin Nikzad-Langerodi et al., "Domain-Invariant Partial Least Squares Regression", Analytical Chemistry, 2018.
    2. Ramin Nikzad-Langerodi et al., "Domain-Invariant Regression under Beer-Lambert's Law", Proc. ICMLA, 2019.
    3. Ramin Nikzad-Langerodi et al., Domain adaptation for regression under Beer–Lambert’s law, Knowledge-Based Systems, 2020.
    4. B. Mikulasek et al., "Partial least squares regression with multiple domains", Journal of Chemometrics, 2023.

    Examples
    --------

    >>> import numpy as np
    >>> from diPLSlib.models import DIPLS
    >>> x = np.random.rand(100, 10)
    >>> y = np.random.rand(100,1)
    >>> xs = np.random.rand(100, 10)
    >>> xt = np.random.rand(50, 10)
    >>> X = np.random.rand(10, 10)
    >>> model = DIPLS(A=5, l=[10])
    >>> model.fit(x, y, xs, xt)
    DIPLS(A=5, l=[10])

    >>> xtest = np.array([5, 7, 4, 3, 2, 1, 6, 8, 9, 10]).reshape(1, -1)
    >>> yhat = model.predict(xtest)
    """

    def __init__(self, A=2, l=[0], centering=True, heuristic=False, target_domain=0, rescale='Target'):
        # Model parameters
        self.A = A
        self.l = l
        self.centering = centering
        self.heuristic = heuristic
        self.target_domain = target_domain
        self.rescale = rescale


    def fit(self, X, y, xs, xt):
        """
        Fit the DIPLS model.

        This method fits the domain-invariant partial least squares (di-PLS) model
        using the provided source and target domain data. It can handle both single 
        and multiple target domains.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Labeled input data from the source domain.

        y : ndarray of shape (n_samples, 1)
            Response variable corresponding to the input data `x`.

        xs : ndarray of shape (n_samples_source, n_features)
            Source domain X-data.

        xt : Union[ndarray of shape (n_samples_target, n_features), List[ndarray]]
            Target domain X-data. Can be a single target domain or a list of arrays 
            representing multiple target domains.


        Returns
        -------
        self : object
            Fitted model instance.
        """

        # Preliminaries
        self.n, self.k = X.shape
        self.ns, _ = xs.shape        
        
        self.x = X
        self.y = y
        self.xs = xs
        self.xt = xt
        self.b0_ = np.mean(self.y)

        # Mean centering
        if self.centering:

            self.mu_ = np.mean(self.x, axis=0)
            self.mu_s_ = np.mean(self.xs, axis=0)
            self.x = self.x - self.mu_
            self.xs = self.xs - self.mu_s_
            y = self.y - self.b0_

            # Mutliple target domains
            if isinstance(self.xt, list):
                
                self.nt, _ = xt[0].shape
                self.mu_t_ = [np.mean(x, axis=0) for x in self.xt]
                self.xt = [x - mu for x, mu in zip(self.xt, self.mu_t_)]
            
            else:

                self.nt, _ = xt.shape
                self.mu_t_ = np.mean(self.xt, axis=0)
                self.xt = self.xt - self.mu_t_

        else:

            y = self.y
        

        x = self.x 
        xs = self.xs
        xt = self.xt

    
        # Fit model
        results = algo.dipals(x, y, xs, xt, self.A, self.l, heuristic=self.heuristic, target_domain=self.target_domain)
        self.b_, self.T_, self.Ts_, self.Tt_, self.W_, self.P_, self.Ps_, self.Pt_, self.E_, self.Es_, self.Et_, self.Ey_, self.C_, self.opt_l_, self.discrepancy_ = results
        
        return self

            
    def predict(self, X):
        """
        Predict y using the fitted DIPLS model.

        This method predicts the response variable for the provided test data using
        the fitted domain-invariant partial least squares (di-PLS) model.

        Parameters
        ----------

        X : ndarray of shape (n_samples, n_features)
            Test data matrix to perform the prediction on.

        Returns
        -------

        yhat : ndarray of shape (n_samples_test,)
            Predicted response values for the test data.

        """
        
        # Rescale Test data 
        if(type(self.rescale) is str):

            if(self.rescale == 'Target'):

                if(type(self.xt) is list):

                    if(self.target_domain==0):

                        Xtest = X[...,:] - self.mu_s_

                    else:

                        Xtest = X[...,:] - self.mu_t_[self.target_domain-1]

                else:

                    Xtest = X[...,:] - self.mu_t_

            elif(self.rescale == 'Source'):

                Xtest = X[...,:] - self.mu_

            elif(self.rescale == 'none'):

                Xtest = X

        elif(type(self.rescale) is np.ndarray):

             Xtest = X[...,:] - np.mean(self.rescale,0)

        else: 

            raise Exception('rescale must either be Source, Target or a Dataset')
            
        
        yhat = Xtest@self.b_ + self.b0_


        return yhat



# Create a separate class for GCT-PLS model inheriting from class model
class GCTPLS(DIPLS):
    """
    Graph-based Calibration Transfer Partial Least Squares (GCT-PLS).

    This method minimizes the distance betwee source (xs) and target (xt) domain data pairs in the latent variable space
    while fitting the response. 

    Parameters
    ----------

    l : Union[int, List[int]], default=0
        Regularization parameter. Can be a single value or a list of different
        values for each latent variable (LV). This parameter controls the degree
        of regularization applied during the fitting process.

    centering : bool, default=True
        If True, source and target domain data are mean-centered before fitting.
        Centering can be crucial in adjusting data for more effective transfer learning.

    heuristic : bool, default=False
        If True, the regularization parameter is set to a heuristic value aimed
        at balancing model fitting quality for the response variable y while minimizing
        discrepancies between domain representations.

    rescale : Union[str, ndarray], default='Target'
        Determines rescaling of the test data. If 'Target' or 'Source', the test data will be rescaled to the mean of xt or xs, respectively. 
        If an ndarray is provided, the test data will be rescaled to the mean of the provided array.


    Attributes
    ----------

    n : int
        Number of samples in `x`.

    ns : int
        Number of samples in `xs`.

    nt : int
        Number of samples in `xt`.

    k : int
        Number of features (variables) in `x`.

    mu_ : ndarray of shape (n_features,)
        Mean of columns in `x`.

    mu_s_ : ndarray of shape (n_features,)
        Mean of columns in `xs`.

    mu_t_ : ndarray of shape (n_features,)
        Mean of columns in `xt`.

    b_ : ndarray of shape (n_features, 1)
        Regression coefficient vector.

    b0_ : float
        Intercept of the regression model.

    T_ : ndarray of shape (n_samples, A)
        Training data projections (scores).

    Ts_ : ndarray of shape (n_source_samples, A)
        Source domain projections (scores).

    Tt_ : ndarray of shape (n_target_samples, A)
        Target domain projections (scores).

    W_ : ndarray of shape (n_features, A)
        Weight matrix.

    P_ : ndarray of shape (n_features, A)
        Loadings matrix corresponding to x.

    Ps_ : ndarray of shape (n_features, A)
        Loadings matrix corresponding to xs.

    Pt_ : ndarray of shape (n_features, A)
        Loadings matrix corresponding to xt.

    E_ : ndarray of shape (n_source_samples, n_features)
        Residuals of source domain data.

    Es_ : ndarray of shape (n_source_samples, n_features)
        Source domain residual matrix.

    Et_ : ndarray of shape (n_target_samples, n_features)
        Target domain residual matrix.

    Ey_ : ndarray of shape (n_source_samples, 1)
        Residuals of response variable in the source domain.

    C_ : ndarray of shape (A, 1)
        Regression vector relating source projections to the response variable.

    opt_l_ : ndarray of shape (A, 1)
        Heuristically determined regularization parameter for each latent variable.

    discrepancy_ : ndarray
        The variance discrepancy between source and target domain projections.


    References
    ----------

    Nikzad‐Langerodi, R., & Sobieczky, F. (2021). Graph‐based calibration transfer. 
    Journal of Chemometrics, 35(4), e3319.

    Examples
    --------
    >>> import numpy as np
    >>> from diPLSlib.models import GCTPLS
    >>> x = np.random.rand(100, 10)
    >>> y = np.random.rand(100, 1)
    >>> xs = np.random.rand(80, 10)
    >>> xt = np.random.rand(80, 10)
    >>> model = GCTPLS(A=3, l=[10])
    >>> model.fit(x, y, xs, xt)
    GCTPLS(A=3, l=[10])
    >>> xtest = np.array([5, 7, 4, 3, 2, 1, 6, 8, 9, 10]).reshape(1, -1)
    >>> yhat = model.predict(xtest)
    """

    def __init__(self, A=2, l=[0], centering=True, heuristic=False, rescale='Target'):
        # Model parameters
        self.A = A
        self.l = l
        self.centering = centering
        self.heuristic = heuristic
        self.rescale = rescale

        
    def fit(self, X, y, xs, xt):
        """
        Fit the GCT-PLS model to data.

        Parameters
        ----------

        x : ndarray of shape (n_samples, n_features)
            Labeled input data from the source domain.

        y : ndarray of shape (n_samples, 1)
            Response variable corresponding to the input data `x`.

        xs : ndarray of shape (n_sample_pairs, n_features)
            Source domain X-data.

        xt : ndarray of shape (n_sample_pairs, n_features)
            Target domain X-data. 
 

        Returns
        -------

        self : object
            Fitted model instance.
        """
        
        # Preliminaries
        self.n, self.k = X.shape
        self.ns, _ = xs.shape
        self.nt, _ = xt.shape

        if self.ns != self.nt:
            raise ValueError("The number of samples in the source domain (ns) must be equal to the number of samples in the target domain (nt).")
        
        self.x = X
        self.y = y
        self.xs = xs
        self.xt = xt
        self.b0_ = np.mean(self.y)
        self.mu_ = np.mean(self.x, axis=0)
        self.mu_s_ = np.mean(self.x, axis=0)
        self.mu_t_ = np.mean(self.x, axis=0)

        # Mean Centering
        if self.centering is True:
            
            x = self.x[...,:] - self.mu_
            y = self.y - self.b0_

        else: 
            
            x = self.x
            y = self.y

        xs = self.xs
        xt = self.xt
            
        # Fit model and store matrices
        results = algo.dipals(x, y, xs, xt, self.A, self.l, heuristic=self.heuristic, laplacian=True)
        self.b_, self.T_, self.Ts_, self.Tt_, self.W_, self.P_, self.Ps_, self.Pt_, self.E_, self.Es_, self.Et_, self.Ey_, self.C_, self.opt_l_, self.discrepancy_ = results

        return self

