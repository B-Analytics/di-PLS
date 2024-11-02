# -*- coding: utf-8 -*-
# Modules
import numpy as np
import matplotlib.pyplot as plt
from diPLSlib import functions as algo
import scipy.stats


class DIPLS:
    """
    Domain-Invariant Partial Least Squares (DIPLS) algorithm for domain adaptation.

    This class implements the DIPLS algorithm, which is designed to align feature distributions 
    across different domains while predicting the target variable `y`. It supports multiple 
    source and target domains through domain-specific feature transformations.

    Parameters
    ----------

    x : ndarray of shape (n_samples, n_features)
        Labeled input data from the source domain.

    y : ndarray of shape (n_samples, 1)
        Response variable corresponding to the input data `x`.

    xs : ndarray of shape (n_samples_source, n_features)
        Source domain feature data.

    xt : Union[ndarray of shape (n_samples_target, n_features), List[ndarray]]
        Target domain feature data. Can be a single target domain or a list of arrays 
        representing multiple target domains.

    A : int
        Number of latent variables to be used in the model.

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

    mu : ndarray of shape (n_features,)
        Mean of columns in `x`.

    mu_s : ndarray of shape (n_features,)
        Mean of columns in `xs`.

    mu_t : ndarray of shape (n_features,) or ndarray of shape (n_domains, n_features)
        Mean of columns in `xt`, averaged per target domain if multiple domains exist.

    b : ndarray of shape (n_features, 1)
        Regression coefficient vector.

    b0 : float
        Intercept of the regression model.

    T : ndarray of shape (n_samples, A)
        Training data projections (scores).

    Ts : ndarray of shape (n_source_samples, A)
        Source domain projections (scores).

    Tt : ndarray of shape (n_target_samples, A)
        Target domain projections (scores).

    W : ndarray of shape (n_features, A)
        Weight matrix.

    P : ndarray of shape (n_features, A)
        Loadings matrix corresponding to x.

    Ps : ndarray of shape (n_features, A)
        Loadings matrix corresponding to xs.

    Pt : ndarray of shape (n_features, A)
        Loadings matrix corresponding to xt.

    E : ndarray of shape (n_source_samples, n_features)
        Residuals of source domain data.

    Es : ndarray of shape (n_source_samples, n_features)
        Source domain residual matrix.

    Et : ndarray of shape (n_target_samples, n_features)
        Target domain residual matrix.

    Ey : ndarray of shape (n_source_samples, 1)
        Residuals of response variable in the source domain.

    C : ndarray of shape (A, 1)
        Regression vector relating source projections to the response variable.

    opt_l : ndarray of shape (A, 1)
        Heuristically determined regularization parameter for each latent variable.

    discrepancy : ndarray
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
    >>> x_test = np.random.rand(10, 10)
    >>> model = DIPLS(x, y, xs, xt, A=5)
    >>> model.fit(l=[0.1], centering=True, heuristic=False)
    >>> yhat, _ = model.predict(x_test, y_test=[], rescale='Target')
    >>> print(yhat)
    """

    def __init__(self, x, y, xs, xt, A):
        self.x = x                         # Labeled X-Data (usually x = xs)
        self.n = np.shape(x)[0]            # Number of X samples
        self.ns = np.shape(xs)[0]          # Number of Xs samples
        self.nt = np.shape(xt)[0]          # Number of Xs samples
        self.k = np.shape(x)[1]            # Number of X variables
        self.y = y                         # Response variable corresponding to X data
        self.xs = xs                       # Source domain X data
        self.xt = xt                       # Target domain X data
        self.mu = np.mean(x, 0)            # Column means of x
        self.mu_s = np.mean(xs, 0)         # Column means of xs
        if(type(self.xt) is list):         # Column Means of xt
            self.ndomains = len(self.xt)   # Multiple Target Domains
            mu_t = np.zeros([self.ndomains, self.k])

            for i in range(self.ndomains):
                mu_t[i, :] = np.mean(self.xt[i], 0)
            self.mu_t = mu_t
        else:
            self.mu_t = np.mean(xt, 0)     # Single Target Domain  
        self.b0 = np.mean(y,0)             # Offset of the response variable
        self.A  = A                        # Number of latent variables


    def fit(self, l=0, centering=True, heuristic=False, target_domain=0):
        """
        Fit the DIPLS model.

        This method fits the domain-invariant partial least squares (di-PLS) model
        using the provided source and target domain data. It can handle both single 
        and multiple target domains.

        Parameters
        ----------

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

        Returns
        -------

        None
        """
        
           
        # Mean Centering
        b0 = np.mean(self.y)
        y = self.y - b0


        if centering is True:

            x = self.x[..., :] - self.mu   
            xs = self.xs[..., :] - self.mu_s


            # Mutliple target domains
            if(type(self.xt) is list):

                xt = [self.xt[i][..., :] - self.mu_t[i, :] for i in range(self.ndomains)]

            else:

                xt = self.xt[..., :] - self.mu_t


        else:

            x = self.x 
            xs = self.xs
            xt = self.xt

    
        # Fit model and store matrices
        A = self.A
        (b, T, Ts, Tt, W, P, Ps, Pt, E, Es, Et, Ey, C, opt_l, discrepancy) = algo.dipals(x, y, xs, xt, A, l, heuristic=heuristic, target_domain=target_domain)
             
        self.b = b
        self.b0 = b0
        self.T = T
        self.Ts = Ts
        self.Tt = Tt
        self.W = W
        self.P = P
        self.Ps = Ps
        self.Pt = Pt
        self.E = E
        self.Es = Es
        self.Et = Et
        self.Ey = Ey
        self.C = C
        self.discrepancy = discrepancy
        self.target_domain = target_domain
        
        
        if heuristic is True:

            self.opt_l = opt_l

            
    def predict(self, x_test, y_test=[], rescale='Target'):
        """
        Predict responses using the fitted DIPLS model.

        This method predicts the response variable for the provided test data using
        the fitted domain-invariant partial least squares (di-PLS) model. It can handle
        rescaling of the test data to match the target domain training set or a provided array.

        Parameters
        ----------

        x_test : ndarray of shape (n_samples_test, n_features)
            Test data matrix to perform the prediction on.

        y_test : ndarray of shape (n_samples_test,), optional
            True response values for the test data. It can be used to evaluate
            the prediction accuracy (default is an empty list).

        rescale : Union[str, ndarray], default='Target'
            Determines rescaling of the test data. If 'Target', the test data will be
            rescaled to the mean of the target domain training set. If an ndarray is provided,
            the test data will be rescaled to the mean of the provided array.

        Returns
        -------

        yhat : ndarray of shape (n_samples_test,)
            Predicted response values for the test data.

        RMSE : float
            Root Mean Squared Error of the predictions if `y_test` is provided.
        """
        
        # Rescale Test data 
        if(type(rescale) is str):

            if(rescale == 'Target'):

                if(type(self.xt) is list):

                    if(self.target_domain==0):

                        Xtest = x_test[...,:] - self.mu_s

                    else:

                        Xtest = x_test[...,:] - self.mu_t[self.target_domain-1, :]

                else:

                    Xtest = x_test[...,:] - self.mu_t

            elif(rescale == 'Source'):

                Xtest = x_test[...,:] - self.mu_s

            elif(rescale == 'none'):

                Xtest = x_test

        elif(type(rescale) is np.ndarray):

             Xtest = x_test[...,:] - np.mean(rescale,0)

        else: 

            raise Exception('rescale must either be Source, Target or a Dataset')
            
        
        yhat = Xtest@self.b + self.b0

        error = algo.rmse(yhat,y_test)


        return yhat,error



# Create a separate class for GCT-PLS model inheriting from class model
class GCTPLS(DIPLS):
    """
    Graph-based Calibration Transfer Partial Least Squares (GCT-PLS).

    This method minimizes the distance betwee source (xs) and target (xt) domain data pairs in the latent variable space
    while fitting the response. 

    Parameters
    ----------

    x : ndarray of shape (n_samples, n_features)
        Labeled input data from the source domain.

    y : ndarray of shape (n_samples, 1)
        Response variable corresponding to the input data `x`.

    xs : ndarray of shape (n_sample_pairs, n_features)
        Source domain feature data.

    xt : ndarray of shape (n_sample_pairs, n_features)
        Target domain feature data. 

    A : int
        Number of latent variables to be used in the model.

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

    mu : ndarray of shape (n_features,)
        Mean of columns in `x`.

    mu_s : ndarray of shape (n_features,)
        Mean of columns in `xs`.

    mu_t : ndarray of shape (n_features,) or ndarray of shape (n_domains, n_features)
        Mean of columns in `xt`, averaged per target domain if multiple domains exist.

    b : ndarray of shape (n_features, 1)
        Regression coefficient vector.

    b0 : float
        Intercept of the regression model.

    T : ndarray of shape (n_samples, A)
        Training data projections (scores).

    Ts : ndarray of shape (n_source_samples, A)
        Source domain projections (scores).

    Tt : ndarray of shape (n_target_samples, A)
        Target domain projections (scores).

    W : ndarray of shape (n_features, A)
        Weight matrix.

    P : ndarray of shape (n_features, A)
        Loadings matrix corresponding to x.

    Ps : ndarray of shape (n_features, A)
        Loadings matrix corresponding to xs.

    Pt : ndarray of shape (n_features, A)
        Loadings matrix corresponding to xt.

    E : ndarray of shape (n_source_samples, n_features)
        Residuals of source domain data.

    Es : ndarray of shape (n_source_samples, n_features)
        Source domain residual matrix.

    Et : ndarray of shape (n_target_samples, n_features)
        Target domain residual matrix.

    Ey : ndarray of shape (n_source_samples, 1)
        Residuals of response variable in the source domain.

    C : ndarray of shape (A, 1)
        Regression vector relating source projections to the response variable.

    opt_l : ndarray of shape (A, 1)
        Heuristically determined regularization parameter for each latent variable.

    discrepancy : ndarray
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
    >>> model = GCTPLS(x, y, xs, xt, 3)
    >>> model.fit(l=[100])
    >>> x_test = np.random.rand(20, 10)
    >>> y_pred, _ = model.predict(x_test)
    >>> print(y_pred)
    """

    def __init__(self, x:np.ndarray, y:np.ndarray, xs:np.ndarray, xt:np.ndarray, A:int=2):
        
        super().__init__(x, y, xs, xt, A)

        
    def fit(self, l=0, centering=True, heuristic=False):
        """
        Fit the GCT-PLS model to data.

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

        Returns
        -------

        self : object
            Fitted model instance. Allows for method chaining in a pipeline setup.
        """
        
        # Mean Centering
        if centering is True:
            
            x = self.x[...,:] - self.mu
            y = self.y - self.b0

        else: 
            
            x = self.x
            y = self.y


        xs = self.xs
        xt = self.xt
            
        # Fit model and store matrices
        A = self.A
        (b, T, Ts, Tt, W, P, Ps, Pt, E, Es, Et, Ey, C, opt_l, discrepancy) = algo.dipals(x, y, xs, xt, A, l, heuristic=heuristic, laplacian=True)

        self.b = b
        self.T = T
        self.Ts = Ts
        self.Tt = Tt
        self.W = W
        self.P = P
        self.Ps = Ps
        self.Pt = Pt
        self.E = E
        self.Es = Es
        self.Et = Et
        self.Ey = Ey
        self.C = C
        self.discrepancy = discrepancy

        if heuristic is True:

            self.opt_l = opt_l


