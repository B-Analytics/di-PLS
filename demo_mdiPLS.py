"""
Copyright Bottleneck Analytics GmbH
Created on Tue Nov 9 12:00:53 2021

@author: Ramin Nikzad-Langerodi

Demonstration of di-PLS with multiple domains

References:
----------

* Nikzad-Langerodi, R., Zellinger, W., Lughofer, E., & Saminger-Platz, S. (2018). 
Domain-invariant partial-least-squares regression. Analytical chemistry, 90(11), 6693-6701.

* Nikzad-Langerodi, R., Zellinger, W., Saminger-Platz, S., & Moser, B. (2019, December).
Domain-invariant regression under Beer-Lambert's law. In 2019 18th IEEE International Conference On Machine Learning And Applications (ICMLA) (pp. 581-586). IEEE.

* Nikzad-Langerodi, R., Zellinger, W., Saminger-Platz, S., & Moser, B. A. (2020). 
Domain adaptation for regression under Beer–Lambert’s law. Knowledge-Based Systems, 210, 106447.

* Nikzad‐Langerodi, R., & Andries, E. 
A chemometrician's guide to transfer learning. Journal of Chemometrics, e3373.

* Bianca Mikulasek, Valeria Fonseca Diaz, David Gabauer, Christoph Herwig, Ramin Nikzad-Langerodi,
Partial least squares regression with multiple domains, J. Chemometrics, 2023 (to appear). 
doi: 10.13140/RG.2.2.23750.75845
"""

#%% Import modules
import numpy as np
import matplotlib.pyplot as plt

import functions as fct
import dipals as ml
from matplotlib.gridspec import GridSpec

#%% Simulate Source and Target Domain data

### Source domain (Analyte + 1 Interferent)
n = 50  # Number of samples
p = 100 # Number of variables

# Generate signals
S1 = fct.gengaus(p, 50, 15, 8, 0)  # Analyte
S2 = fct.gengaus(p, 70, 10, 10, 0) # Interferent
S = np.vstack([S1,S2])

# Analyte concentrations
Cs = 10*np.random.rand(n,S.shape[0])

# Spectra
X = Cs@S

# Random noise
noise = 0.005*np.random.rand(n,p)

# Source domain spectra plus noise
Xs = X + noise

# Target domain 1 (Analyte + 2 Interferents)
S1 = fct.gengaus(p, 50, 15, 8, 0)  # Analyte
S2 = fct.gengaus(p, 70, 10, 10, 0) # Interferent 1
S3 = fct.gengaus(p, 30, 10, 10, 0) # Interferent 2
S = np.vstack([S1,S2,S3])

# Analyte concentrations
Ct1 = 10*np.random.rand(n,S.shape[0])

# Spectra
X = Ct1@S

# Random noise
noise = 0.005*np.random.rand(n,p)

# Target domain spectra plus noise
Xt1 = X# + noise

# Target domain 2 (Analyte + 3 Interferents)
S1 = fct.gengaus(p, 50, 15, 8, 0)  # Analyte
S2 = fct.gengaus(p, 70, 10, 10, 0) # Interferent 1
S3 = fct.gengaus(p, 30, 10, 10, 0) # Interferent 2
S4 = fct.gengaus(p, 75, 75, 150, 0) # Interferent 3
S = np.vstack([S1, S2, S3, S4])

# Analyte concentrations
Ct2 = 10*np.random.rand(n,S.shape[0])

# Spectra
X = Ct2@S

# Random noise
noise = 0.005*np.random.rand(n,p)

# Target domain spectra plus noise
Xt2 = X# + noise

plt.figure(figsize=(12,5))
plt.subplot(211)
plt.plot(S1)
plt.plot(S2)
plt.plot(S3)
plt.plot(S4)
plt.legend(['Analyte','Interferent 1','Interferent 2', 'Interferent 3'])
plt.title('Pure Signals')
plt.xlabel('X-Variables')
plt.ylabel('Signal')
plt.axvline(x=50,linestyle='-',color='k',alpha=0.5)
plt.axvline(x=70,linestyle=':',color='k',alpha=0.5)
plt.axvline(x=30,linestyle=':',color='k',alpha=0.5)


plt.subplot(234)
plt.plot(Xs.T, 'b', alpha=0.2)
plt.title('Source Domain')
plt.xlabel('X-Variables')
plt.ylabel('Signal')

plt.subplot(235)
plt.plot(Xt1.T, 'r', alpha=0.2)
plt.title('Target Domain 1')
plt.xlabel('X-Variables')
plt.ylabel('Signal')

plt.subplot(236)
plt.plot(Xt2.T, 'g', alpha=0.2)
plt.title('Target Domain 2')
plt.xlabel('X-Variables')
plt.ylabel('Signal')
plt.tight_layout()
plt.show()

#%% Projections from a PCA and a mmdi-PLS Model

### PCA
X = np.vstack([Xs, Xt1, Xt2])
X = X[:,...] - np.mean(X, 0)
U,S,V = np.linalg.svd(X)
T = U[:, :100]@np.diag(S)

plt.figure(figsize=(9, 3))
plt.subplot(121)
a = plt.scatter(T[:50, 0], T[:50, 1], edgecolors='k')
el = fct.hellipse(T[:50, :2])
plt.plot(el[0,:],el[1,:])

b = plt.scatter(T[51:100, 0], T[51:100, 1], edgecolors='k')
el = fct.hellipse(T[51:100, :2])
plt.plot(el[0,:],el[1,:])

c = plt.scatter(T[101:, 0], T[101:, 1], edgecolors='k')
el = fct.hellipse(T[101:, :2])
plt.plot(el[0,:],el[1,:])

plt.xlabel('PC 1')
plt.ylabel('PC 2')

ax = plt.gca()
ax.axhline(y=0,color='k',linestyle=':')
ax.axvline(x=0,color='k',linestyle=':')

plt.legend([a, b, c], ['Source', 'Target 1', 'Target 2'])
plt.title('PCA')


### mdi-PLS
nr_comp = 2
l =  [100]
target_domains = [Xt1, Xt2]
ys = np.expand_dims(Cs[:, 0],1)
yt1 = np.expand_dims(Ct1[:, 0],1)
yt2 = np.expand_dims(Ct2[:, 0],1)

m_dipls = ml.model(Xs, ys, Xs, target_domains, 2)
m_dipls.fit(l=l, target_domain=1)

plt.subplot(122)
a = plt.scatter(m_dipls.Ts[:, 0], m_dipls.Ts[:, 1], edgecolors='k')
el = fct.hellipse(m_dipls.Ts)
plt.plot(el[0,:],el[1,:])

b = plt.scatter(m_dipls.Tt[0][:, 0], m_dipls.Tt[0][:, 1], edgecolors='k')
el = fct.hellipse(m_dipls.Tt[0])
plt.plot(el[0,:],el[1,:])

c = plt.scatter(m_dipls.Tt[1][:, 0], m_dipls.Tt[1][:, 1], edgecolors='k')
el = fct.hellipse(m_dipls.Tt[1])
plt.plot(el[0,:],el[1,:])

ax = plt.gca()
ax.axhline(y=0,color='k',linestyle=':')
ax.axvline(x=0,color='k',linestyle=':')

plt.xlabel('LV 1')
plt.ylabel('LV 2')
plt.title('mdi-PLS')
plt.legend([a, b, c], ['Source', 'Target 1', 'Target 2'])

plt.tight_layout()
plt.show()

#%% Source domain model and domain-invariant model applied to target domains
# Note that invariant models are derived in an unsupervised fashion 
# (i.e. no concentration values are used from the target domains)

### Source model
m_pls = ml.model(Xs, ys, Xs, target_domains, nr_comp)
m_pls.fit(l=[0], target_domain=2)
b_pls = m_pls.b

yhat_plsT1, error_plsT1 = m_pls.predict(Xt1, yt1)
yhat_plsT2, error_plsT2 = m_pls.predict(Xt2, yt2)

### mmdi-PLS Model for target domain 1
m_diplsT1 = ml.model(Xs, ys, Xs, target_domains, nr_comp)
m_diplsT1.fit(l=l, target_domain=1)
b_diplsT1 = m_diplsT1.b
yhat_diplsT1, error_diplsT1 = m_diplsT1.predict(Xt1, yt1)

### mmdi-PLS Model for target domain 2
m_diplsT2 = ml.model(Xs, ys, Xs, target_domains, nr_comp)
m_diplsT2.fit(l=l, target_domain=2)
b_diplsT2 = m_diplsT2.b
yhat_diplsT2, error_diplsT2 = m_diplsT2.predict(Xt2, yt2)

min_ = np.min(np.array([yt1,yt2,yhat_plsT1,yhat_plsT2]))
max_ = np.max(np.array([yt1,yt2,yhat_plsT1,yhat_plsT2]))

# Plot
plt.figure(figsize=(9, 3))
plt.subplot(121)
plt.scatter(yt1, yhat_plsT1, color='b', edgecolor='k',alpha=0.75)
plt.scatter(yt1, yhat_diplsT1, color='m', edgecolor='k',alpha=0.75)
plt.legend(['PLS', 'mdi-PLS'])
plt.plot([min_,max_], [min_,max_], color='k', linestyle=":")
plt.xlim([min_,max_])
plt.ylim([min_,max_])
plt.title('Predictions in Target Domain 1')
plt.xlabel('Measured')
plt.ylabel('Predicted')

plt.subplot(122)
plt.scatter(yt2, yhat_plsT2, color='b', edgecolor='k',alpha=0.75)
plt.scatter(yt2, yhat_diplsT2, color='m', edgecolor='k',alpha=0.75)
plt.legend(['PLS', 'mdi-PLS'])
plt.plot([min_,max_], [min_,max_], color='k', linestyle=":")
plt.xlim([min_,max_])
plt.ylim([min_,max_])
plt.title('Predictions in Target Domain 2')
plt.xlabel('Measured')
plt.ylabel('Predicted')


#%% Regression coefficients, weights and loadings
fig = plt.figure(figsize=(9, 9), constrained_layout=True)
gs = GridSpec(4, 3, figure=fig)

# create sub plots as grid
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1], sharey=ax4)
ax5_1 = fig.add_subplot(gs[1, 2], sharey=ax4)
ax6 = fig.add_subplot(gs[2, 0])
ax7 = fig.add_subplot(gs[2, 1:], sharey=ax6)
ax8 = fig.add_subplot(gs[3, 0])
ax9 = fig.add_subplot(gs[3, 1], sharey=ax8)
ax10 = fig.add_subplot(gs[3, 2], sharey=ax8)

# Raw data
ax1.plot(Xs.T, 'b', alpha=0.2)
ax1.set_title('Source Domain')
ax1.set_xlabel('X-Variables')
ax1.set_ylabel('Signal')

ax2.plot(Xt1.T, 'r', alpha=0.2)
ax2.set_title('Target Domain 1')
ax2.set_xlabel('X-Variables')
ax2.set_ylabel('Signal')

ax3.plot(Xt2.T, 'g', alpha=0.2)
ax3.set_title('Target Domain 2')
ax3.set_xlabel('X-Variables')
ax3.set_ylabel('Signal')

# Regression coefficients
ax4.plot(b_pls, 'b')
ax4.set_ylabel('Reg Coeffs.')
ax4.set_xlabel('X-Variables')
ax4.set_title('PLS Model (Source Domain)')
ax4.axhline(y=0, Color='k')

ax5.plot(b_diplsT1, 'r')
ax5.set_ylabel('Reg Coeffs.')
ax5.set_xlabel('X-Variables')
ax5.set_title('mdi-PLS Model (Target 1 specific)')
ax5.axhline(y=0, Color='k')

ax5_1.plot(b_diplsT2, 'g')
ax5_1.set_ylabel('Reg Coeffs.')
ax5_1.set_xlabel('X-Variables')
ax5_1.set_title('mdi-PLS Model (Target 2 specific)')
ax5_1.axhline(y=0, Color='k')

# Weights
ax6.plot(m_pls.W[:, 0], 'b')
ax6.plot(m_pls.W[:, 1], '--', Color='b')
ax6.set_ylabel('Weights')
ax6.legend(['LV1','LV2'])
ax6.set_xlabel('X-Variables')
ax6.set_title('PLS Model (Source Domain)')
ax6.axhline(y=0, Color='k')

ax7.plot(m_diplsT1.W[:, 0],'m')
ax7.plot(m_diplsT1.W[:, 1], '--', Color='m')
ax7.set_ylabel('Weights')
ax7.legend(['LV1','LV2'])
ax7.set_xlabel('X-Variables')
ax7.set_title('mdi-PLS Model (All Domains)')
ax7.axhline(y=0, Color='k')

# Loadings
ax8.plot(m_pls.Ps[:, 0], 'b')
ax8.plot(m_pls.Ps[:, 1], '--', Color='b')
ax8.set_ylabel('Loadings')
ax8.legend(['LV1','LV2'])
ax8.set_xlabel('X-Variables')
ax8.set_title('PLS Model (Source Domain)')
ax8.axhline(y=0, Color='k')

ax9.plot(m_diplsT1.Pt[0][:, 0],'r')
ax9.plot(m_diplsT1.Pt[0][:, 1], '--', Color='r')
ax9.set_ylabel('Loadings')
ax9.legend(['LV1','LV2'])
ax9.set_xlabel('X-Variables')
ax9.set_title('mdi-PLS Model (Target 1 specific)')
ax9.axhline(y=0, Color='k')

ax10.plot(m_diplsT2.Pt[1][:, 0], 'g')
ax10.plot(m_diplsT2.Pt[1][:, 1], '--', Color='g')
ax10.set_ylabel('Loadings')
ax10.legend(['LV1','LV2'])
ax10.set_xlabel('X-Variables')
ax10.set_title('mdi-PLS Model (Target 2 specific)')
ax10.axhline(y=0, Color='k')

plt.tight_layout()
plt.show()
