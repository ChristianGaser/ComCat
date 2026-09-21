### ComCat: Theoretical framework

#### Data model

ComCat extends the location/scale model underlying ComBat (Johnson et al., 2007; Fortin et al., 2017) by partitioning the design matrix into three components. For feature $v$ ($v = 1, \ldots, P$, e.g., voxels or brain regions) and sample $j$ ($j = 1, \ldots, N$), the model is:

$$Y_{jv} = \mathbf{B}_j\boldsymbol{\beta}_{\text{batch},v} + f_Z(\mathbf{Z}_j)_v + \mathbf{X}_j\boldsymbol{\beta}_{X,v} + \varepsilon_{jv}$$

where $\mathbf{B}$ is the $N \times K$ dummy-coded batch indicator matrix ($K$ = number of sites; reduced to a single intercept column when no site information is provided), $\mathbf{Z}$ is the $N \times Q$ matrix of continuous nuisance covariates (e.g., IQMs), $f_Z$ denotes a smooth nonlinear function of $\mathbf{Z}$ realized through B-spline basis expansion (see below), and $\mathbf{X}$ is the $N \times R$ matrix of covariates of interest to be preserved (e.g., age, group). The batch indicators $\mathbf{B}$ and nuisance covariates $\mathbf{Z}$ together represent the unwanted variation to be removed, while $\mathbf{X}$ contains the biological effects to be preserved.


#### B-spline expansion of continuous nuisance variables

Each continuous nuisance variable $z_q$ is expanded into a B-spline basis of dimension $d$:

$$z_q \mapsto \tilde{\mathbf{Z}}_q = [b_1(z_q),\ b_2(z_q),\ \ldots,\ b_d(z_q)]$$

where $b_1, \ldots, b_d$ are cubic B-spline basis functions with interior knots placed at quantiles of $z_q$ and boundary knots at its observed range (or at `smooth_term_bounds`, if given). The basis dimension is the standard GAM smoothness parameter, denoted `gam_df` throughout this paper. To balance flexibility and overfitting risk while adapting to sample size, `gam_df` is selected automatically as:

$${gam_{df}} = \min\left(10,\ \max\left(5,\ \lfloor N/30 \rfloor\right)\right)$$

The lower bound of 5 ensures that nonlinear shape can be captured even in small samples; the upper bound of 10 limits flexibility to prevent overfitting. The implications of this upper bound for preserving biological group differences are addressed in the Discussion. The expanded nuisance design matrix is denoted $\tilde{\mathbf{Z}}$ with $Q' = Q \cdot d$ columns.


#### Parameter estimation

All parameters are estimated jointly in a single ordinary least squares (OLS) step on the full design matrix:

$$\hat{\boldsymbol{\beta}} = \text{pinv}\left(\bigl[\mathbf{B}\ \tilde{\mathbf{Z}}\ \mathbf{X}\bigr]\right) \cdot Y^T$$

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

$$\hat{\gamma}_v = \text{pinv}\left([\mathbf{B}\ \tilde{\mathbf{Z}}]\right) \cdot (Y_v^{\text{std}})^T$$

The site-specific multiplicative (variance) effects are estimated directly from the within-site variance of the standardized data:

$$\hat{\delta}_{i,v}^2 = \text{Var}\left( Y_{v,\text{batch}=i}^{\text{std}} \right)$$

Estimating $\hat{\delta}_{i,v}^2$ from the within-site variance directly — without first removing the additive nuisance effects — preserves the full per-site variance structure that the multiplicative correction is intended to remove. If the mean-only option is set, $\hat{\delta}_{i,v}^2 = 1$ for all batches.

> **Note.** Because $\hat{\delta}_{i,v}^2$ includes the variance explained by the nuisance effects, dividing the nuisance-adjusted residuals by it also reduces the residual variance itself. The optional `residual_delta` estimator (see *Optional extensions*) estimates $\hat{\delta}_{i,v}^2$ after removing the nuisance effects, as ComBat does.


#### Data adjustment

For each batch $i$, the adjustment is:

$$Y_{v,\text{batch}=i}^{\text{adj}} = \frac{Y_{v,\text{batch}=i}^{\text{std}} - [\mathbf{B}\ \tilde{\mathbf{Z}}]_{\text{batch}=i}\hat{\gamma}_v}{\sqrt{\hat{\delta}_{i,v}^2}}$$

The subtraction term removes both the batch-specific additive shift and the smooth nuisance contributions simultaneously, since the full $\hat{\gamma}_v$ vector is applied via the combined design matrix $[\mathbf{B}\ \tilde{\mathbf{Z}}]$.


#### Rescaling to original units

The adjusted data are rescaled back to the original data space:

$$Y_{jv}^{\text{ComCat}} = Y_{jv}^{\text{adj}} \cdot \text{RMSE}_v + \bar{\mu}_v + \mathbf{X}_j\hat{\boldsymbol{\beta}}_{X,v}$$

This restores the original data scale while preserving the biological effects encoded in $\mathbf{X}$.


#### Optional extensions

Three extensions can be enabled in `comcat()` / `comcat_ui()`. All are off by default; the default output is identical to the model described above.

**1. Flexible preserved covariates (`preserve_df`).** Each continuous preserved covariate $x_r$ (at least `preserve_df` + 2 distinct values) is expanded into the same kind of B-spline basis as the nuisance covariates, $x_r \mapsto \tilde{\mathbf{X}}_r = [b_1(x_r), \ldots, b_d(x_r)]$; binary or categorical covariates stay linear. Estimation is unchanged, with $\tilde{\mathbf{X}}$ in place of $\mathbf{X}$ in the joint model, and the preserved effect $\tilde{\mathbf{X}}_j\hat{\boldsymbol{\beta}}_{\tilde{X},v}$ is added back.

*Rationale.* With a linear $\mathbf{X}$ and a spline-expanded $\tilde{\mathbf{Z}}$, the model is more flexible for the nuisance than for the effects of interest. If $\mathbf{Z}$ depends non-linearly on $\mathbf{X}$ (for example, image quality is worse in children and older adults), functions in the span of $\tilde{\mathbf{Z}}$ can represent the non-linear part of the effect of $\mathbf{X}$. The joint OLS then attributes that part to $\tilde{\mathbf{Z}}$, and it is removed. With $\tilde{\mathbf{X}}$ in the model, the non-linear effect of interest is fitted explicitly and the nuisance coefficients are estimated conditional on it. In a simulation with a lifespan-like age curve and seven IQMs that are U-shaped functions of age, the linear age term lost ~70–80% of the age curve, whereas with `preserve_df` the loss was ~4–5% (`tests/test_comcat_options.py`).

**2. Site variances from residuals (`residual_delta`).** Write the standardized data of site $i$ as $Y^{\text{std}} = \gamma_i + \tilde{\mathbf{Z}}\hat{\gamma}_Z + e$. The default estimator is

$$\hat{\delta}_{i,v}^2 = \text{Var}_i\left(Y_v^{\text{std}}\right) \approx \text{Var}_i\left(\tilde{\mathbf{Z}}\hat{\gamma}_{Z,v}\right) + \text{Var}_i\left(e_v\right),$$

so the adjusted residual $e_v / \hat{\delta}_{i,v}$ has variance

$$\frac{\text{Var}_i(e_v)}{\text{Var}_i(\tilde{\mathbf{Z}}\hat{\gamma}_{Z,v}) + \text{Var}_i(e_v)} \approx 1 - R^2_{Z,i,v}.$$

That is, the residual variance of the harmonized data shrinks by the in-sample share of within-site variance explained by the nuisance terms. This share grows with `gam_df`, and it differs between sites with different nuisance spread. `comcat_ui()` codes a missing site as one site, so single-site analyses are affected as well; only `mean_only` avoids the effect. With `residual_delta=True`,

$$\hat{\delta}_{i,v}^2 = \text{Var}_i\left(Y_v^{\text{std}} - [\mathbf{B}\ \tilde{\mathbf{Z}}]\hat{\gamma}_v\right),$$

which matches ComBat, where the site variance is computed after all covariate effects are removed. The residual variance is then preserved and equalized across sites.

**3. Nuisance-dependent variance (`nuisance_scale`).** The additive model assumes that nuisance covariates affect only the mean. Image quality can also change the noise level (heteroscedasticity). With `nuisance_scale=True`, the residuals $e_{jv}$ (after the additive site and nuisance effects are removed) are modelled as

$$e_{jv} \sim N\left(0, \sigma_{jv}^2\right), \qquad \log \sigma_{jv} = \mathbf{B}_j\boldsymbol{\theta}_{B,v} + \mathbf{z}^{*}_j\boldsymbol{\theta}_{Z,v} + \mathbf{x}^{*}_j\boldsymbol{\theta}_{X,v},$$

where $\mathbf{z}^{*}$ and $\mathbf{x}^{*}$ are the z-scored raw nuisance and preserved covariates, entered linearly. The parameters are estimated for every feature by maximum likelihood: Fisher scoring with working weights 2, as for the $\sigma$ parameter in GAMLSS, with step halving. The nuisance-dependent part of the scale is then removed,

$$e^{*}_{jv} = e_{jv}\,\exp\left(-\mathbf{z}^{*}_j\hat{\boldsymbol{\theta}}_{Z,v}\right),$$

which rescales every subject to the noise level at the mean nuisance value ($\mathbf{z}^{*} = 0$). Site and preserved-covariate terms stay in the variance model. Site variance differences are therefore not attributed to the nuisance covariates, and they are removed afterwards by $\hat{\delta}$, now estimated from $e^{*}$ (this option implies `residual_delta`). Biological effects on the variance are retained.

The same fit provides a per-feature likelihood-ratio test of nuisance effects on the variance: $\Lambda_v = D_{0,v} - D_{1,v}$, the deviance difference between the models without and with $\mathbf{z}^{*}$, which is asymptotically $\chi^2_Q$ under $H_0: \boldsymbol{\theta}_{Z,v} = 0$ (returned as `scale_lr` and `scale_p`).

The variance model is the one of ComBatLS (Gardner et al., 2025), used for the opposite purpose. ComBatLS models covariate effects on the scale in order to *preserve* them; here, nuisance effects on the scale are modelled in order to *remove* them, while preserved-covariate effects on the scale are kept.

`comcat_from_training()` applies all three extensions to new data with the training estimates (spline knots, $\hat{\boldsymbol{\theta}}_Z$, and the z-scoring of the training data).

## References

Fortin JP, Parker D, Tunç B, Watanabe T, Elliott MA, Ruparel K, et al. (2017). Harmonization of multi-site diffusion tensor imaging data. *NeuroImage*, 161, 149–170.

Gardner M, Shinohara RT, Bethlehem RAI, Romero-Garcia R, Warrier V, Dorfschmidt L, et al. (2025). ComBatLS: A location- and scale-preserving method for multi-site image harmonization. *Human Brain Mapping*, 46(8), e70197.

Johnson WE, Li C, Rabinovic A (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*, 8(1), 118–127.

