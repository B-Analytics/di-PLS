# -*- coding: utf-8 -*-

# Modules
import numpy as np
import scipy.linalg
import scipy.stats
from scipy.linalg import eigh
import scipy.spatial.distance as scd
from scipy.spatial import distance_matrix
import warnings
warnings.filterwarnings("ignore")
from sklearn.utils.validation import check_array
from diPLSlib.utils.misc import calibrateAnalyticGaussianMechanism


def dipals(x, y, xs, xt, A, l, heuristic: bool = False, target_domain=0, laplacian: bool = False):
    """
    Perform (Multiple) Domain-Invariant Partial Least Squares (di-PLS) regression.

    This method fits a PLS regression model using labeled source domain data and potentially 
    unlabeled target domain data across multiple domains, aiming to build a model that 
    generalizes well across different domains.

    Parameters
    ----------

    x : ndarray of shape (n_samples, n_features)
        Labeled source domain data.

    y : ndarray of shape (n_samples, 1)
        Response variable associated with the source domain.

    xs : ndarray of shape (n_source_samples, n_features)
        Source domain feature data.

    xt : ndarray of shape (n_target_samples, n_features) or list of ndarray
        Target domain feature data. Multiple domains can be provided as a list.

    A : int
        Number of latent variables to use in the model.

    l : float or tuple of len(l)=A
        Regularization parameter. If a single value is provided, the same regularization is applied to all latent variables.

    heuristic : bool, default=False
        If True, automatically determine the regularization parameter to equally balance fitting 
        to Y and minimizing domain discrepancy.

    target_domain : int, default=0
        Specifies which target domain the model should apply to, where 0 indicates the source domain.

    laplacian : bool, default=False
        If True, uses a Laplacian matrix to regularize distances between matched calibration transfer 
        samples in latent variable space.

    Returns
    -------

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
    >>> from diPLSlib.functions import dipals
    >>> x = np.random.random((100, 10))
    >>> y = np.random.random((100, 1))
    >>> xs = np.random.random((50, 10))
    >>> xt = np.random.random((50, 10))
    >>> b, T, Ts, Tt, W, P, Ps, Pt, E, Es, Et, Ey, C, opt_l, discrepancy = dipals(x, y, xs, xt, 2, (0.1))
    """

    # Get array dimensions
    (n, k) = np.shape(x)
    (ns, k) = np.shape(xs)

    
    # Initialize matrices
    Xt = xt

    if(type(xt) is list):
        Pt = []
        Tt = []


        for z in range(len(xt)):

                Tti = np.zeros([np.shape(xt[z])[0], A])
                Pti = np.zeros([k, A])

                Pt.append(Pti)
                Tt.append(Tti)


    else:

        (nt, k) = np.shape(xt)
        Tt = np.zeros([nt, A])
        Pt = np.zeros([k, A])


    T = np.zeros([n, A])
    P = np.zeros([k, A])
    Ts = np.zeros([ns, A])
    Ps = np.zeros([k, A])
    W = np.zeros([k, A])
    C = np.zeros([A, 1])
    opt_l = np.zeros(A)
    discrepancy = np.zeros(A)
    I = np.eye(k)

    # Compute LVs
    for i in range(A):

        if isinstance(l, tuple) and len(l) == A:       # Separate regularization params for each LV

            lA = l[i]

        elif isinstance(l, (float, int, np.int64)):              # The same regularization param for each LV

            lA = l

        else:

            raise ValueError("The regularization parameter must be either a single value or an A-tuple.")


        # Compute Domain-Invariant Weight Vector
        w_pls = ((y.T@x)/(y.T@y))  # Ordinary PLS solution
       


        if(lA != 0 or heuristic is True):  # In case of regularization

                if(type(xt) is not list):

                    # Convex relaxation of covariance difference matrix
                    D = convex_relaxation(xs, xt)

                # Multiple target domains
                elif(type(xt) is list):
                    
                    #print('Relaxing domains ... ')
                    ndoms = len(xt)
                    D = np.zeros([k, k])

                    for z in range(ndoms):

                        d = convex_relaxation(xs, xt[z])
                        D = D + d

                elif(laplacian is True):
                
                    J = np.vstack([xs, xt])
                    L = transfer_laplacian(xs, xt)
                    D = J.T@L@J


                else:

                    print('xt must either be a matrix or list of (appropriately dimensioned) matrices')

                if(heuristic is True): # Regularization parameter heuristic

                    w_pls = w_pls/np.linalg.norm(w_pls)
                    gamma = (np.linalg.norm((x-y@w_pls))**2)/(w_pls@D@w_pls.T)
                    opt_l[i] = gamma
                    lA = gamma


                reg = I+lA/((y.T@y))*D
                w = scipy.linalg.solve(reg.T, w_pls.T, assume_a='sym').T  # 10 times faster than previous comptation of reg

                # Normalize w
                w = w/np.linalg.norm(w)

                # Absolute difference between variance of source and target domain projections
                discrepancy[i] = (w @ D @ w.T).item()


        else:        

            if(type(xt) is list):

                D = convex_relaxation(xs, xt[0])

            else:

                D = convex_relaxation(xs, xt)

            
            w = w_pls/np.linalg.norm(w_pls)
            discrepancy[i] = (w @ D @ w.T).item()

    
        # Compute scores
        t = x@w.T
        ts = xs@w.T
        
        if(type(xt) is list):

            tt = []

            for z in range(len(xt)):

                tti = xt[z]@w.T
                tt.append(tti)

        else:

            tt = xt@w.T


        # Regress y on t
        c = (y.T@t)/(t.T@t)

        # Compute loadings
        p = (t.T@x)/(t.T@t)
        ps = (ts.T@xs)/(ts.T@ts)
        if(type(xt) is list):

            pt = []

            for z in range(len(xt)):

                pti = (tt[z].T@xt[z])/(tt[z].T@tt[z])
                pt.append(pti)

        else:

            pt = (tt.T@xt)/(tt.T@tt)


        # Deflate X and y (Gram-Schmidt orthogonalization)
        x = x - t@p

        if laplacian is False:                       # Calibration transfer case
            xs = xs - ts@ps
        
        if(type(xt) is list):

            for z in range(len(xt)):

                xt[z] = xt[z] - tt[z]@pt[z]

        else:

            if(np.sum(xt) != 0):  # Deflate target matrix only if not zero

                if laplacian is False:                       # Calibration transfer case
                    xt = xt - tt@pt


        y = y - t*c

        # Store w,t,p,c
        W[:, i] = w
        T[:, i] = t.reshape(n)
        Ts[:, i] = ts.reshape(ns)
        P[:, i] = p.reshape(k)
        Ps[:, i] = ps.reshape(k)
        C[i] = c       

        if(type(xt) is list):

            for z in range(len(xt)):

                Pt[z][:, i] = pt[z].reshape(k)
                Tt[z][:, i] = tt[z].reshape(np.shape(xt[z])[0])

        else:
            
            Pt[:, i] = pt.reshape(k)
            Tt[:, i] = tt.reshape(nt)         


    # Calculate regression vector
    if laplacian is True:                       # Calibration transfer case

        b = W@(np.linalg.inv(P.T@W))@C

    else:

        if isinstance(l, tuple):                # Check if multiple regularization # parameters are passed (one for each LV)

            if target_domain==0:                # Multiple target domains (Domain unknown)

                b = W@(np.linalg.inv(P.T@W))@C

            elif type(xt) is np.ndarray:        # Single target domain

                b = W@(np.linalg.inv(Pt.T@W))@C

            elif type(xt) is list:              # Multiple target domains (Domain known)

                b = W@(np.linalg.inv(Pt[target_domain-1].T@W))@C

        else:

                b = W@(np.linalg.inv(P.T@W))@C   


    # Store residuals
    E = x
    Es = xs
    Et = xt
    Ey = y

    return b, T, Ts, Tt, W, P, Ps, Pt, E, Es, Et, Ey, C, opt_l, discrepancy


def convex_relaxation(xs, xt):
    """
    Perform convex relaxation of the covariance difference matrix.

    This relaxation involves computing the eigenvalue decomposition of the symmetric covariance 
    difference matrix, inverting the signs of negative eigenvalues, and reconstructing the matrix.
    This corresponds to an upper bound on the covariance difference between source and target domains.

    Parameters
    ----------

    xs : ndarray of shape (n_source_samples, n_features)
        Feature data from the source domain.

    xt : ndarray of shape (n_target_samples, n_features)
        Feature data from the target domain.

    Returns
    -------

    D : ndarray of shape (n_features, n_features)
        Relaxed covariance difference matrix.

    References
    ----------

    Ramin Nikzad-Langerodi et al., "Domain-Invariant Regression under Beer-Lambert's Law", Proc. ICMLA, 2019.

    Examples
    --------

    >>> import numpy as np
    >>> from diPLSlib.functions import convex_relaxation
    >>> xs = np.random.random((100, 10))
    >>> xt = np.random.random((100, 10))
    >>> D = convex_relaxation(xs, xt)
    """

    # Ensure input arrays are numerical
    xs = np.asarray(xs, dtype=np.float64)
    xt = np.asarray(xt, dtype=np.float64)
    
    # Check for NaN or infinite values
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(xt)):
        raise ValueError("Input arrays must not contain NaN or infinite values. one sample.")

    # Check for complex data
    if np.iscomplexobj(xs) or np.iscomplexobj(xt):
        raise ValueError("Complex data not supported.")
    
    # Preliminaries
    ns = np.shape(xs)[0]
    nt = np.shape(xt)[0]
    x = np.vstack([xs, xt])
    x = x[..., :] - np.mean(x, 0)
    
    # Compute difference between source and target covariance matrices   
    rot = (1/ns*xs.T@xs- 1/nt*xt.T@xt) 

    # Convex Relaxation
    w,v = eigh(rot)
    eigs = np.abs(w)
    eigs = np.diag(eigs)
    D = v@eigs@v.T 

    return D
                    

def transfer_laplacian(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Construct a Laplacian matrix for calibration transfer problems.

    Parameters
    ----------

    x : ndarray of shape (n_samples, n_features)
        Data samples from device 1.

    y : ndarray of shape (n_samples, n_features)
        Data samples from device 2.

    Returns
    -------
    L : ndarray of shape (2 * n_samples, 2 * n_samples)
        The Laplacian matrix for the calibration transfer problem.

    References
    ----------

    Nikzad‐Langerodi, R., & Sobieczky, F. (2021). Graph‐based calibration transfer. 
    Journal of Chemometrics, 35(4), e3319.

    Examples
    --------

    >>> import numpy as np
    >>> from diPLSlib.functions import transfer_laplacian
    >>> x = np.array([[1, 2], [3, 4]])
    >>> y = np.array([[2, 3], [4, 5]])
    >>> L = transfer_laplacian(x, y)
    >>> print(L)
    [[ 1.  0. -1. -0.]
     [ 0.  1. -0. -1.]
     [-1. -0.  1.  0.]
     [-0. -1.  0.  1.]]
    """

    (n, p) = np.shape(x)
    I = np.eye(n)
    L = np.vstack([np.hstack([I,-I]),np.hstack([-I,I])])

    return L


def edpls(x:np.ndarray, y:np.ndarray, n_components:int, epsilon:float, delta:float=0.05):
    '''
    :math:`(\epsilon, \delta)`-Differentially Private Partial Least Squares Regression.

    A Gaussian mechanism according to Balle & Wang (2018) is used to privately release weights :math:`\mathbf{W}`, scores :math:`\mathbf{T}`
    and :math:`X/Y`-loadings :math:`\mathbf{P}`/:math:`\mathbf{c}` from the PLS1 algorithm. To this end, for each latent variable, i.i.d. noise from 
    :math:`\mathcal{N}(0,\sigma^2)` with variance satisfying
    
    .. math::
           \Phi\left( \\frac{\Delta}{2\sigma} - \\frac{\epsilon\sigma}{\Delta} \\right) - e^{\epsilon} \Phi\left( -\\frac{\Delta}{2\sigma} - \\frac{\epsilon\sigma}{\Delta} \\right)\leq \delta,

    with :math:`\Phi(t) = \mathrm P[\mathcal{N}(0,1)\leq t]` (i.e., the CDF of the standard univariate Gaussian distribution), is added to the weights, scores and loadings, whereas the sensitivity :math:`\Delta(\cdot)` for the functions releasing the corresponding quantities is calculated as follows:

    .. math::
        \Delta(w) = \sup_{(\mathbf{x}, y)} |y| \|\mathbf{x}\|_2

    .. math::
        \Delta(t) \leq \sup_{\mathbf{x}}  \|\mathbf{x}\|_2

    .. math::
        \Delta(p) \leq \sup_{\mathbf{x}}  \|\mathbf{x}\|_2

    .. math::
        \Delta(c) \leq \sup_{y}  |y|.
    
    Note that in contrast to the Gaussian mechanism, proposed in Dwork et al. (2006) and Dwork et al. (2014), the mechanism of Balle & Wang (2018) guarantees :math:`(\epsilon, \delta)`-differential privacy 
    for any value of :math:`\epsilon > 0` and not only for :math:`\epsilon \leq 1`.

    Parameters
    ----------

    x: numpy array of shape (n, m)
        x-data

    y: numpy array of shape (n, p)


    n_components: int
        Number of latent variables.

    epsilon: float
        Privacy loss parameter.

    delta: float, default=0.05
        Failure probability.


    Returns
    -------

    coef_: numpy array of shape (m,)
        Regression coefficients.

    x_scores_: numpy array of shape (n, A)
        X scores.

    x_loadings_: numpy array of shape (m, A)
        X loadings.

    x_weights_: numpy array of shape (m, A)
        X weights.

    y_loadings_: numpy array of shape (A, )
        Y loadings.

    x_residuals: numpy array of shape (n, m)
        X residuals.

    y_residuals: numpy array of shape (n, p)
        Y residuals.

    
    References
    ----------

    - R.Nikzad-Langerodi, et al. (2024). (epsilon,delta)-Differentially private partial least squares regression (2024, unpublished).
    - Balle, B., & Wang, Y. X. (2018, July). Improving the gaussian mechanism for differential privacy: Analytical calibration and optimal denoising. In International Conference on Machine Learning (pp. 394-403). PMLR.

    Examples
    --------

    >>> from diPLSlib.functions import edpls
    >>> import numpy as np
    >>> x = np.random.rand(100, 10)
    >>> y = np.random.rand(100, 1)
    >>> coef_, x_weights_, x_loadings_, y_loadings_, x_scores_, x_residuals_, y_residuals_ = edpls(x, y, 2, epsilon=0.1, delta=0.05)
    '''


    # Get dimensions of arrays
    (n_, n_features_) = np.shape(x)
    I = np.eye(n_features_)

    # Weights
    x_weights_ = np.zeros([n_features_, n_components])

    # X Scores
    x_scores_ = np.zeros([n_, n_components])

    # X Loadings
    x_loadings_ = np.zeros([n_features_, n_components])

    # Y Loadings
    y_loadings_ = np.zeros([n_components, 1])

    # Iterate over the number of components
    for i in range(n_components):

        # Compute weights w
        w_pls = ((y.T@x)/(y.T@y))  

        # Normalize w (noise-less)
        wo = w_pls / np.linalg.norm(w_pls)

        # Compute x scores and normalize (noise-less)
        to = x @ wo.T
        to = to / np.linalg.norm(to)

        # Compute x scores and normalize (before adding noise)
        t = x @ wo.T
        t = to / np.linalg.norm(to)

        # Add noise to w
        x_min = x.min(axis=0)
        x_max = x.max(axis=0)
        y_min = y.min(axis=0)
        y_max = y.max(axis=0)
        x_norm = np.linalg.norm(x_max - x_min)
        y_norm = y_max - y_min
        x_max_norm = np.linalg.norm(x, axis=1).max()
        
        sensitivity = x_max_norm*y_max
        R = calibrateAnalyticGaussianMechanism(epsilon, delta, sensitivity)**2
        
        v = np.random.normal(0, R, n_features_)
        w = wo + v

        # Normalize w (after adding noise)
        w = w / np.linalg.norm(w)

        # Add noise to t
        sensitivity = x_max_norm
        R = calibrateAnalyticGaussianMechanism(epsilon, delta, sensitivity)**2
        v = np.random.normal(0, R, n_)
        t = t + v.reshape(n_,1)

        # Normalize t (after adding noise)
        t = t / np.linalg.norm(t)

        # Compute x loadings (noise-less)
        po = (to.T@x)/(to.T@to)

        # Compute x loadings (before adding noise)
        p = (to.T@x)/(to.T@to)

        # Add noise
        #sensitivity = 2*x_max_norm
        sensitivity = x_max_norm
        R = calibrateAnalyticGaussianMechanism(epsilon, delta, sensitivity)**2
        v = np.random.normal(0, R, n_features_)
        p = p + v

        # Compute y loadings (noise-less)
        co = (y.T@to)/(to.T@to)

        # Compute y loadings (before adding noise)
        c = (y.T@to)/(to.T@to)

        # Add noise
        sensitivity = y_max
        R = calibrateAnalyticGaussianMechanism(epsilon, delta, sensitivity)**2
        v = np.random.normal(0, R, 1)
        c = c + v

        # Store weights, scores and loadings
        x_weights_[:, i] = w
        x_scores_[:, i] = t.reshape(n_)
        x_loadings_[:, i] = p.reshape(n_features_)
        y_loadings_[i] = c

        # Deflate x and y
        x = x - to @ po
        y = y - to * co

    # Compute regression coefficients
    coef_ = x_weights_@(np.linalg.inv(x_loadings_.T@x_weights_))@y_loadings_

    # Compute residuals
    x_residuals_ = x
    y_residuals_ = y

    return (coef_, x_weights_, x_loadings_, y_loadings_, x_scores_, x_residuals_, y_residuals_ )


