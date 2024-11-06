# Domain-invariant partial least squares regression (di-PLS)

![](https://img.shields.io/badge/python-3.13-blue.svg)


Python implementation of (m)di-PLS for domain adaptation in multivariate regression problems. 

![](https://user-images.githubusercontent.com/77445667/104728864-d5fede80-5737-11eb-8aad-59f9901a0cf4.png)

# Installation
```bash
pip install diPLSlib
```

# Quick Start
## How to apply di-PLS
Train regression model
```python
from diPLSlib.models import DIPLS
from diPLSlib.utils import misc

l = 100000                    # or l = (10000, 100) Regularization
m = DIPLS(A=2, l=l)
m.fit(X, y, X_source, X_target)

# Typically X=X_source and y are the corresponding response values
```
Apply the model 
```python
yhat_dipls = m.predict(X_test)
err = misc.rmse(y_test, yhat_dipls)
```

## How to apply mdi-PLS
```python
from diPLSlib.models import DIPLS

l = 100000                    # or l = (5, 100, 1000)  Regularization
m = DIPLS(A=3, l=l, target_domain=2)
m.fit(X, y, X_source, X_target)

# X_target = [X1, X2, ... , Xk] is a list of target domain data
# The parameter target_domain specifies for which domain the model should be trained (here X2).
```

## How to apply GCT-PLS
```python
from diPLSlib.models import GCTPLS

# Training
l = 10                         # or l = (10, 10) Regularization
m = GCTPLS(A=2, l=l)
m.fit(X, y, X_source, X_target)

# X_source and X_target hold the same samples measured in the source and target domain, respectively.
```

## Examples
For more examples, please refer to the [Notebooks](notebooks):

- [Domain adaptation with di-PLS](notebooks/diPLS_example.ipynb)
- [Including multiple domains (mdi-PLS)](notebooks/mdiPLS_example.ipynb)
- [Implicit calibration transfer with GCT-PLS](notebooks/GCTPLS_example.ipynb)
- [Model selection with `scikit-learn`](notebooks/demo_ModelSelection_SciKitLearn.ipynb)

# Documentation
The documentation can be found [here](https://di-pls.readthedocs.io/en/latest/diPLSlib.html).

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


