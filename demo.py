"""
Bottleneck Analytics GmbH
info@bottleneck-analytics.com

@author: Ramin Nikzad-Langerodi

Application of Domain-Invariant Partial Least Squares
(di-PLS) regression on a simulated data set for finding a
good model that generalizes over a Source and a Target domain.

"""

#%% Import modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import functions as fct
import dipals as ml

#%% Simulate Source and Target Domain Data
np.random.seed(10)

### Source domain (Analyte + 1 Interferent)
n = 50  # Number of samples
p = 100 # Number of variables

# Generate signals
S1 = fct.gengaus(p, 50, 15, 8, 0)  # Analyte
S2 = fct.gengaus(p, 70, 10, 10, 0) # Interferent

S = np.vstack([S1,S2])

# Analyte and Interferent concentrations
Cs = 10*np.random.rand(n,2)

# Spectra
Xs = Cs@S

### Target domain (Analyte + 2 Interferents)
S1 = fct.gengaus(p, 50, 15, 8, 0)  # Analyte
S2 = fct.gengaus(p, 70, 10, 10, 0) # Interferent 1
S3 = fct.gengaus(p, 30, 10, 10, 0) # Interferent 2

S = np.vstack([S1,S2,S3])

# Analyte and interferent concentrations
Ct = 10*np.random.rand(n,3)

# Spectra
Xt = Ct@S

### Plot data
plt.figure(figsize=(9,5))

plt.subplot(211)
plt.plot(S1)
plt.plot(S2)
plt.plot(S3)
plt.legend(['Analyte','Interferent 1','Interferent 2'])
plt.title('Pure Signals')
plt.xlabel('X-Variables')
plt.ylabel('Signal')
plt.axvline(x=50,linestyle='-',color='k',alpha=0.5)
plt.axvline(x=70,linestyle=':',color='k',alpha=0.5)
plt.axvline(x=30,linestyle=':',color='k',alpha=0.5)

plt.subplot(223)
plt.plot(Xs.T, 'b', alpha=0.2)
plt.title('Source Domain')
plt.xlabel('X-Variables')
plt.ylabel('Signal')
plt.axvline(x=50,linestyle='-',color='k',alpha=0.5)
plt.axvline(x=70,linestyle=':',color='k',alpha=0.5)

plt.subplot(224)
plt.plot(Xt.T, 'r', alpha=0.2)
plt.title('Target Domain')
plt.xlabel('X-Variables')
plt.ylabel('Signal')
plt.axvline(x=50,linestyle='-',color='k',alpha=0.5)
plt.axvline(x=70,linestyle=':',color='k',alpha=0.5)
plt.axvline(x=30,linestyle=':',color='k',alpha=0.5)
plt.tight_layout()

#%% Source domain Model and Domain-invariant Model
# Prepare plots
f = plt.figure(figsize=(9,5))
spec = f.add_gridspec(2,2)

gs0 = spec[:,0].subgridspec(2,1)
gs1 = spec[0,1].subgridspec(2,2)
gs2 = spec[1,1]


# Spectra plots
ax1 = f.add_subplot(gs0[0,0])
ax2 = f.add_subplot(gs0[1,0])

ax1.plot(Xs.T, 'b', alpha=0.2)
ax1.plot(np.mean(Xs.T,1),'b')
ax2.plot(Xt.T, 'r', alpha=0.15)
ax2.plot(np.mean(Xt.T,1),'r')

ax1.axvline(x=50,linestyle='-',color='k',alpha=0.5)
ax1.axvline(x=68,linestyle=':',color='k',alpha=0.5)
ax2.axvline(x=50,linestyle='-',color='k',alpha=0.5)
ax2.axvline(x=68,linestyle=':',color='k',alpha=0.5)
ax2.axvline(x=32,linestyle=':',color='k',alpha=0.5)

ax1.set_ylabel('Signal')
ax2.set_ylabel('Signal')
ax1.set_xlabel('X-Variables')
ax2.set_xlabel('X-Variables')

ax1.set_title('Source Domain')
ax2.set_title('Target Domain')

# Scores and regression coefficients plots
ax3 = f.add_subplot(gs1[0,0])
ax4 = f.add_subplot(gs1[0,1])
ax5 = f.add_subplot(gs1[1,0])
ax6 = f.add_subplot(gs1[1,1])

# Measured vs predicted plot
ax7 = f.add_subplot(gs2)

# Source domain PLS Models (2 LVs)
y = np.expand_dims(Cs[:, 0],1)
m = ml.model(Xs, y, Xs, Xt, 2)
l = [0] # No regularization
m.fit(l)
b_source = m.b
yhat_pls, err = m.predict(Xt, Ct[:, 0])

ax5.plot(b_source, 'b')
ax5.axhline(y=0, linestyle='-',color='k')
ax5.set_ylabel('Reg. Coefs.')
ax5.set_xlabel('X-Variables')
ax5.tick_params(labelleft=False, left=False)
ax5.axvline(x=50,linestyle='-',color='k',alpha=0.5)
ax5.axvline(x=70,linestyle=':',color='k',alpha=0.5)

ax3.axhline(y=0,color='k',linestyle=':')
ax3.axvline(x=0,color='k',linestyle=':')

ax3.plot(m.Ts[:, 0], m.Ts[:, 1], '.b', MarkerEdgecolor='k', alpha=0.75)
el = fct.hellipse(m.Ts)
ax3.plot(el[0,:],el[1,:],'b')
ax3.plot(m.Tt[:, 0], m.Tt[:, 1], '.r', MarkerEdgecolor='k', alpha=0.75)
el = fct.hellipse(m.Tt)
ax3.plot(el[0,:],el[1,:],'r')

ax3.tick_params(labelleft=False, labelbottom=False, bottom=False, left=False)
ax3.set_title('Source PLS')
ax3.set_xlabel('LV 1')
ax3.set_ylabel('LV 2')

# di-PLS Model (2 LVs)
y = np.expand_dims(Cs[:, 0],1)
m = ml.model(Xs, y, Xs, Xt, 2)
l = 10000
m.fit(l=l)
b_not = m.b
yhat_dipls, err = m.predict(Xt, Ct[:, 0])

ax6.plot(b_not, 'm')
ax6.axhline(y=0, linestyle='-',color='k')
ax6.set_xlabel('X-Variables')
ax6.tick_params(labelleft=False, left=False)
ax6.axvline(x=30,linestyle=':', color='k', alpha=0.5)
ax6.axvline(x=50,linestyle='-', color='k', alpha=0.5)
ax6.axvline(x=70,linestyle=':', color='k', alpha=0.5)

ax4.axhline(y=0,color='k',linestyle=':')
ax4.axvline(x=0,color='k',linestyle=':')
ax4.plot(m.Ts[:, 0], m.Ts[:, 1], '.b', MarkerEdgecolor='k', alpha=0.75)
el = fct.hellipse(m.Ts)
ax4.plot(el[0,:],el[1,:],'b')
ax4.plot(m.Tt[:, 0], m.Tt[:, 1], '.r', MarkerEdgecolor='k', alpha=0.75)
el = fct.hellipse(m.Tt)
ax4.plot(el[0,:],el[1,:],'r')
ax4.tick_params(labelleft=False, labelbottom=False, bottom=False, left=False)
ax4.set_title('di-PLS')
ax4.set_xlabel('LV 1')

ax7.scatter(Ct[:, 0], yhat_pls, color='b', edgecolor='k',alpha=0.75)
ax7.scatter(Ct[:, 0], yhat_dipls, color='m', edgecolor='k',alpha=0.75)
ax7.legend(['PLS','di-PLS'])
ax7.plot([-5,14],[-5,14],'k',linestyle='-')
ax7.set_xlim([0,10])
ax7.set_ylim([-5,15])
ax7.set_xlabel('Measured')
ax7.set_ylabel('Predicted')
ax7.grid(axis='x')

plt.tight_layout()
