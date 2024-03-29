# Domain-invariant partial least squares regression (di-PLS)

Python implementation of di-PLS for domain adaptation in multivariate regression problems. 

![demo](https://user-images.githubusercontent.com/77445667/104728864-d5fede80-5737-11eb-8aad-59f9901a0cf4.png)

## How to apply di-PLS
Train regression model
```python
import dipals as ml

m = ml.model(X, y, X_source, X_target, 2)
l = 100000 #  Regularization
m.fit(l)

# Typically X=X_source and y are the corresponding response values
```
Apply the model 
```python
yhat_dipls, err = m.predict(X_test, y_test=[])

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
Partial least squares regression with multiple domains, J. Chemometrics, 2023 (to appear), https://doi.org/10.13140/RG.2.2.23750.75845*

# Contact us
Bottleneck Analytics GmbH  
info@bottleneck-analytics.com


