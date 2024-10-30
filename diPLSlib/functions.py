# -*- coding: utf-8 -*-
"""
Bottleneck Analytics GmbH
info@bottleneck-analytics.com

@author: Dr. Ramin Nikzad-Langerodi
"""

# Modules
import numpy as np
import scipy.linalg
import scipy.stats
from scipy.linalg import eigh
import scipy.spatial.distance as scd
from scipy.spatial import distance_matrix
import warnings
warnings.filterwarnings("ignore")


def dipals(x, y, xs, xt, A, l, heuristic:bool=False, target_domain=0, laplacian:bool=False):
        '''
        (Multiple) Domain-invariant partial least squares regression ((m)di-PLS) performs PLS regression 
        using labeled Source domain data x (=xs) and y, and unlabeled Target domain data (xt_1,...,xt_k)
        with the goal to fit an (invariant) model that generalizes over all domains.


        References:
        (1) Ramin Nikzad-Langerodi, Werner Zellinger, Edwin Lughofer, and Susanne Saminger-Platz 
          "Domain-Invariant Partial Least Squares Regression" Analytical Chemistry 2018 90 (11), 
          6693-6701 DOI: 10.1021/acs.analchem.8b00498

        (2) Ramin Nikzad-Langerodi, Werner Zellinger, Susanne Saminger-Platz and Bernhard Moser 
          "Domain-Invariant Regression under Beer-Lambert's Law" In Proc. International Conference
          on Machine Learning and Applications, Boca Raton FL 2019.

        (3) Ramin Nikzad-Langerodi, Werner Zellinger, Susanne Saminger-Platz, Bernhard A. Moser, 
          Domain adaptation for regression under Beer–Lambert’s law, Knowledge-Based Systems, 
          2020 (210) DOI: 10.1016/j.knosys.2020.106447

        (4) B. Mikulasek, V. Fonseca Diaz, D. Gabauer, C. Herwig, and R. Nikzad-Langerodi
            "Partial least squares regression with multiple domains" Journal of Chemometrics 2023 37 (5), e3477

            
        Parameters
        ----------
        x: numpy array (N,K)
            Labeled X data
            
        y: numpy array (N,1)
            Response variable
        
        xs: numpy array (N_S,K) 
            Source domain data
        
        xt: numpy array (N_T, K) or list of z (N_z, K) numpy arrays
            Target domain data. If list is passed multiple target domains are considered
            in the optimization
            
        A: int
            Number of latent variables
            
        l: int or numpy array (A x 1)
            Regularization parameter: Typically 0 < l < 1e10
            If Array is passed, a different l is used for each LV
            
        heuristic: str
            If 'True' the regularization parameter is determined using a 
            heuristic that gives equal weight to: 
            i) Fitting the output variable y and 
            ii) Minimizing domain discrepancy.
            For details see ref. (3).

        target_domain: int
            If multiple target domains are passed, target_domain specifies for which of the target domains
            the model should apply. If target_domain=0, the model applies to the source domain,
            if target_domain=1, the model applies to the first target domain etc.

        laplacian: bool
            If True, the Laplacian matrix obtained by calling the transfer_laplacian function is used
            to regularize the distance between matched calibration transfer samples in the LV space.
            
        Returns
        -------
        b: numpy array (K,1)
            Regression vector
            
        b0: int
            Offset (Note: yhat = b0 + x*b)
        
        T: numpy array (N,A)
            Training data projections (scores)
        
        Ts: numpy array (N_S,A)
            Source domain projections (scores)
            
        Tt: numpy array (N_T,A)
            Target domain projections (scores)
        
        W: numpy array (K,A)
            Weight vector
            
        P: numpy array (K,A)
            Loadings vector
            
        E: numpy array (N_S,K)
            Residuals of labeled X data   
            
        Es: numpy array (N_S,K)
            Source domain residual matrix
            
        Et: numpy array (N_T,K)
            Target domain residual matrix
            
        Ey: numpy array (N_S,1)
            Response variable residuals
            
        C: numpy array (A,1)
            Regression vector, such that
            y = Ts*C
        
        opt_l: numpy array (A,1)
            The heuristically determined regularization parameter for each LV 
            (if heuristic = 'True')
            
        discrepancy: numpy array (A,)
            Absolute difference between variance of source and target domain projections
        '''

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

            if type(l) is np.ndarray:  # Separate regularization params for each LV

                lA = l[i]

            elif(type(l) is int or type(l) is np.float64): # One regularization for all LVs

                lA = l

            else:

                lA = l[0]


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
                 discrepancy[i] = w@D@w.T


            else:        

                if(type(xt) is list):

                    D = convex_relaxation(xs, xt[0])

                else:

                    D = convex_relaxation(xs, xt)

                
                w = w_pls/np.linalg.norm(w_pls)
                discrepancy[i] = w@D@w.T

        
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

            if np.any([i != 0 for i in l]):         # Check if multiple regularization # parameters are passed (one for each LV)

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
        '''
        Convex relaxation of covariance difference.
         
        The convex relaxation computes the eigenvalue decomposition of the (symetric) covariance 
        difference matrix, inverts the signs of the negative eigenvalues and reconstructs it again. 
        It can be shown that this relaxation corresponds to an upper bound on the covariance difference
        btw. source and target domain (see ref.)

        
        Reference:

        * Ramin Nikzad-Langerodi, Werner Zellinger, Susanne Saminger-Platz and Bernhard Moser 
          "Domain-Invariant Regression under Beer-Lambert's Law" In Proc. International Conference
          on Machine Learning and Applications, Boca Raton FL 2019.
        
        Parameters
        ----------
        
        xs: numpy array (Ns x k)
            Source domain matrix
            
        xt: numpy array (Nt x k)
            Target domain matrix
            
        Returns
        -------
        
        D: numpy array (k x k)
            Covariance difference matrix
        
        '''
        
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
                    

def gengaus(length, mu, sigma, mag, noise=0):
    '''
    Generate a spectrum-like Gaussian signal with random noise

    Params
    ------

    length: int
        Length of the signal (i.e. number of variables)

    mu: float
        Mean of the Gaussian signal

    sigma: float
        Standard deviation of the Gaussian

    mag: float
        Magnitude of the signal

    noise: float
        Amount of i.i.d noise


    Returns
    -------

    signal: numpy array (length x 1)
        Generated Gaussian signal
    '''

    s = mag*scipy.stats.norm.pdf(np.arange(length),mu,sigma)
    n = noise*np.random.rand(length)
    signal = s + n

    return signal


def hellipse(X, alpha=0.05): 
    '''
    95% Confidence Intervals for 2D Scatter Plots
     
    Parameters
    ----------    
    X: numpy array (N x 2)
        Scores Matrix

    alpha: float
        Confidence level (default = 0.05)
               
    Returns
    -------
    el: numpy array (2 x 100)
       x,y coordinates of ellipse points arranged in rows. 
       To plot use plt.plot(el[0,:],el[1,:])     
    '''  
    
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
    Conf = (2*(N-2))/(N-2)*scipy.stats.f.ppf(quant,2,(N-2))
    
    # Evalute CI on (0,2pi)
    el = np.zeros((2,100))
    t = np.linspace(0,2*np.pi,100)
    for j in np.arange(100):
        sT = np.matmul(U,np.diag(np.sqrt(S*Conf)))
        el[:,j] = np.transpose(mean_all)+np.matmul(sT,np.array([np.cos(t[j]),np.sin(t[j])]))   

    return el


def rmse(y, yhat):
    '''
    Root mean squared error
        
    Parameters
    ----------    
    y: numpy array (N,1)
            Measured Y
    
    yhat: numpy array (N,1)
        Predicted Y    
        
    Returns
    -------
    int
        The Root Means Squared Error
    '''  

    return np.sqrt(((y-yhat)**2).mean())


def transfer_laplacian(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    """
    Laplacian matrix for calibration transfer problems.

    Laplacian matrix L = [[-1, 1];[1, -1]] x I, where I is the identity matrix, for dataset J = [xs;xt], where s.t. xs and xt
    contain the matched calibration transfer standard samples.

    See: Nikzad‐Langerodi, R., & Sobieczky, F. (2021). Graph‐based calibration transfer. Journal of Chemometrics, 35(4), e3319.   

    Parameters
    ----------
    x: numpy array (N x K)
        Calibration transfer samples device 1

    y: numpy array (N x K)
        Calibration transfer samples device 2


    Returns:
    --------
    L: numpy array (2N x 2N)
        Laplacian matrix

    """

    (n, p) = np.shape(x)
    I = np.eye(n)
    L = np.vstack([np.hstack([I,-I]),np.hstack([-I,I])])

    return L

