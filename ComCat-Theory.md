### ComCat: Theoretical framework

#### Data model

ComCat extends the location/scale model underlying ComBat (Johnson et al., 2007; Fortin et al., 2017) by partitioning the design matrix into three components. For feature $v$ ($v = 1, \ldots, P$, e.g., voxels or brain regions) and sample $j$ ($j = 1, \ldots, N$), the model is:

$$Y_{jv} = \mathbf{B}_j\boldsymbol{\beta}_{\text{batch},v} + f_Z(\mathbf{Z}_j)_v + \mathbf{X}_j\boldsymbol{\beta}_{X,v} + \varepsilon_{jv}$$

where $\mathbf{B}$ is the $N \times K$ dummy-coded batch indicator matrix ($K$ = number of sites; reduced to a single intercept column when no site information is provided), $\mathbf{Z}$ is the $N \times Q$ matrix of continuous nuisance covariates (e.g., IQMs), $f_Z$ denotes a smooth nonlinear function of $\mathbf{Z}$ realized through B-spline basis expansion (see below), and $\mathbf{X}$ is the $N \times R$ matrix of covariates of interest to be preserved (e.g., age, group). The batch indicators $\mathbf{B}$ and nuisance covariates $\mathbf{Z}$ together represent the unwanted variation to be removed, while $\mathbf{X}$ contains the biological effects to be preserved.


#### B-spline expansion of continuous nuisance variables

Each continuous nuisance variable $z_q$ is expanded into a B-spline basis of dimension $d$:

$$z_q \mapsto \tilde{\mathbf{Z}}_q = [b_1(z_q),\ b_2(z_q),\ \ldots,\ b_d(z_q)]$$

where $b_1, \ldots, b_d$ are B-spline basis functions defined over equally spaced knots within the observed range of $z_q$. The basis dimension is the standard GAM smoothness parameter, denoted `gam_df` throughout this paper. To balance flexibility and overfitting risk while adapting to sample size, `gam_df` is selected automatically as:

$${gam_{df}} = \min\left(10,\ \max\left(5,\ \lfloor N/30 \rfloor\right)\right)$$

The lower bound of 5 ensures that nonlinear shape can be captured even in small samples; the upper bound of 10 limits flexibility to prevent overfitting. The implications of this upper bound for preserving biological group differences are addressed in the Discussion. The expanded nuisance design matrix is denoted $\tilde{\mathbf{Z}}$ with $Q' = Q \cdot d$ columns.


#### Parameter estimation

All parameters are estimated jointly in a single ordinary least squares (OLS) step on the full design matrix:

$$\hat{\boldsymbol{\beta}} = \text{pinv}\!\left(\bigl[\mathbf{B}\ \tilde{\mathbf{Z}}\ \mathbf{X}\bigr]\right) \cdot Y^T$$

By the Frisch–Waugh–Lovell theorem, the estimate $\hat{\boldsymbol{\beta}}_X$ for the preserved covariates is identical whether obtained from this joint model or from a regression on residuals after partialling out $[\mathbf{B}\ \tilde{\mathbf{Z}}]$. The preserved covariates are therefore correctly accounted for regardless of their correlation with batch or nuisance variables, provided the design matrix is not rank-deficient.


#### Standardization

The grand mean is defined as the sample-average predicted value from the batch and nuisance components:

$$\bar{\mu}_v = \frac{1}{N} \sum_{j=1}^{N} \bigl([\mathbf{B}\ \tilde{\mathbf{Z}}]_j \hat{\boldsymbol{\beta}}_{[\text{batch},Z],v}\bigr)$$

The pooled standard deviation is the root mean squared error of the full-model residuals:

$$\text{RMSE}_v = \sqrt{\frac{1}{N} \sum_{j=1}^{N} \bigl(Y_{jv} - [\mathbf{B}\ \tilde{\mathbf{Z}}\ \mathbf{X}]_j \hat{\boldsymbol{\beta}}_v\bigr)^2}$$

The standardized data are obtained by removing the grand mean and the preserved covariate effects, then dividing by the pooled standard deviation:

$$Y_v^{\text{std}} = \frac{Y_v - \bar{\mu}_v - \mathbf{X}\hat{\boldsymbol{\beta}}_{X,v}}{\text{RMSE}_v}$$


#### Estimation of nuisance effects on standardized data

After standardization, the additive effects of both batch and nuisance variables are re-estimated from the standardized data using the reduced design matrix $[\mathbf{B}\ \tilde{\mathbf{Z}}]$:

$$\hat{\gamma}_v = \text{pinv}\!\left([\mathbf{B}\ \tilde{\mathbf{Z}}]\right) \cdot (Y_v^{\text{std}})^T$$

The site-specific multiplicative (variance) effects are estimated directly from the within-site variance of the standardized data:

$$\hat{\delta}_{i,v}^2 = \text{Var}\!\left( Y_{v,\text{batch}=i}^{\text{std}} \right)$$

Estimating $\hat{\delta}_{i,v}^2$ from the within-site variance directly — without first removing the additive nuisance effects — preserves the full per-site variance structure that the multiplicative correction is intended to remove. If the mean-only option is set, $\hat{\delta}_{i,v}^2 = 1$ for all batches.


#### Data adjustment

For each batch $i$, the adjustment is:

$$Y_{v,\text{batch}=i}^{\text{adj}} = \frac{Y_{v,\text{batch}=i}^{\text{std}} - [\mathbf{B}\ \tilde{\mathbf{Z}}]_{\text{batch}=i}\hat{\gamma}_v}{\sqrt{\hat{\delta}_{i,v}^2}}$$

The subtraction term removes both the batch-specific additive shift and the smooth nuisance contributions simultaneously, since the full $\hat{\gamma}_v$ vector is applied via the combined design matrix $[\mathbf{B}\ \tilde{\mathbf{Z}}]$.


#### Rescaling to original units

The adjusted data are rescaled back to the original data space:

$$Y_{jv}^{\text{ComCat}} = Y_{jv}^{\text{adj}} \cdot \text{RMSE}_v + \bar{\mu}_v + \mathbf{X}_j\hat{\boldsymbol{\beta}}_{X,v}$$

This restores the original data scale while preserving the biological effects encoded in $\mathbf{X}$.

## References

Fortin JP, Parker D, Tunç B, Watanabe T, Elliott MA, Ruparel K, et al. (2017). Harmonization of multi-site diffusion tensor imaging data. *NeuroImage*, 161, 149–170.

Johnson WE, Li C, Rabinovic A (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*, 8(1), 118–127.

