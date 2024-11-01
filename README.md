# Domain-invariant partial least squares regression (di-PLS)

![](https://img.shields.io/badge/python-3.13-blue.svg)


Python implementation of (m)di-PLS for domain adaptation in multivariate regression problems. 

![](https://user-images.githubusercontent.com/77445667/104728864-d5fede80-5737-11eb-8aad-59f9901a0cf4.png)

# Installation
```bash
pip install diPLSlib
```

# Usage 
## How to apply di-PLS
Train regression model
```python
from diPLSlib.models import DIPLS as dipls

m = dipls(X, y, X_source, X_target, 2)
l = [100000] #  Regularization
m.fit(l)

# Typically X=X_source and y are the corresponding response values
```
Apply the model 
```python
yhat_dipls, err = m.predict(X_test, y_test=[])
```

## How to apply mdi-PLS
```python
from diPLSlib.models import DIPLS as dipls

# Training
m = dipls(X, y, X_source, X_target, 2)
l = [100000] #  Regularization
m.fit(l, target_domain=2)

# Testing
yhat_dipls, err = m.predict(X_test, y_test=[])


# X_target = [X1, X2, ... , Xk] is a list of target domain data
# The parameter target_domain specifies for which domain the model should be trained (here X2).
```

## How to apply GCT-PLS
```python
from diPLSlib.models import GCTPLS as gctpls

# Training
m = gctpls(X, y, X_source, X_target, 2)
l = [100] #  Regularization
m.fit(l)

# Testing
yhat_gct, err = m.predict(X_test, y_test=[])


# X_source and X_target hold the same samples measured in the source and target domain, respectively.
```

# Acknowledgements
The first version of di-PLS was developed by Ramin Nikzad-Langerodi, Werner Zellinger, Edwin Lughofer, Bernhard Moser and Susanne Saminger-Platz
and published in:

- *Ramin Nikzad-Langerodi, Werner Zellinger, Edwin Lughofer, and Susanne Saminger-Platz
Analytical Chemistry 2018 90 (11), 6693-6701 https://doi.org/10.1021/acs.analchem.8b00498*

Further refinements to the initial algorithm were published in: 

- *R. Nikzad-Langerodi, W. Zellinger, S. Saminger-Platz and B. Moser, "Domain-Invariant Regression Under Beer-Lambert's Law," 2019 18th IEEE International Conference On Machine Learning And Applications (ICMLA), Boca Raton, FL, USA, 2019, pp. 581-586, https://doi.org/10.1109/ICMLA.2019.00108.*

- *Ramin Nikzad-Langerodi, Werner Zellinger, Susanne Saminger-Platz, Bernhard A. Moser,
Domain adaptation for regression under Beer–Lambert’s law,
Knowledge-Based Systems, Volume 210, 2020, https://doi.org/10.1016/j.knosys.2020.106447.*

- *Bianca Mikulasek, Valeria Fonseca Diaz, David Gabauer, Christoph Herwig, Ramin Nikzad-Langerodi,
"Partial least squares regression with multiple domains" Journal of Chemometrics 2023 37 (5), e3477, https://doi.org/10.13140/RG.2.2.23750.75845*

- *Ramin Nikzad-Langerodi & Florian Sobieczky (2021). Graph‐based calibration transfer. Journal of Chemometrics, 35(4), e3319. https://doi.org/10.1002/cem.3319*

# Contact us
Bottleneck Analytics GmbH  
info@bottleneck-analytics.com


