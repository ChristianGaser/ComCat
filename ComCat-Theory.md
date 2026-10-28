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

> **Note.** The basis is not penalized. With the seven CAT12 IQMs, which are partly redundant (SIQR ≈ res_ECR; IQR is largely a combination of NCR, ICR and res_RMS), the design is rank-deficient or near-singular. The fit then does not generalize to held-out scans. IQM effects estimated across subjects also partly reflect anatomy. See [HeldOut-Evaluation.md](HeldOut-Evaluation.md).


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


#### Empirical Bayes estimation of site effects

With at least two sites, the site parameters are by default (`eb=True`) estimated with the parametric empirical Bayes method of ComBat (Johnson et al., 2007). With few subjects per site, the L/S estimates $\hat{\gamma}_{i,v}$ and $\hat{\delta}_{i,v}^2$ are noisy, and dividing by a variance estimated from a handful of values can amplify the data of single features many times. Empirical Bayes borrows information across features.

The site rows of $\hat{\gamma}_v$ are the site levels at the zero of the nuisance basis relative to the grand mean. They therefore share a per-feature offset that belongs to the nuisance model and differs between features. Shrinking them directly would leave a site-dependent fraction of this offset in the data. They are first expressed as differences to the size-weighted mean of the sites,

$$\hat{\gamma}^{c}_{i,v} = \hat{\gamma}_{i,v} - \sum_{k} \frac{n_k}{N}\,\hat{\gamma}_{k,v},$$

which are the site effects of ComBat when the nuisance covariates are part of its standardization model. With a reference site $r$, the differences to site $r$ are used. For every site, the effects of all features are assumed to come from common priors,

$$\gamma^{c}_{i,v} \sim N\left(\bar{\gamma}_i, \bar{\tau}_i^2\right), \qquad \delta_{i,v}^2 \sim \text{InvGamma}\left(\lambda_i, \theta_i\right),$$

whose hyperparameters are estimated by the method of moments from $\hat{\gamma}^{c}_{i,v}$ and $\hat{\delta}_{i,v}^2$ over all features. The posterior means are found by iterating

$$\gamma^{*}_{i,v} = \frac{n_i\bar{\tau}_i^2\,\hat{\gamma}^{c}_{i,v} + \delta^{2*}_{i,v}\,\bar{\gamma}_i}{n_i\bar{\tau}_i^2 + \delta^{2*}_{i,v}}, \qquad \delta^{2*}_{i,v} = \frac{\theta_i + \frac{1}{2}\left[(n_i - 1)\,\hat{\delta}_{i,v}^2 + n_i\left(\hat{\gamma}^{c}_{i,v} - \gamma^{*}_{i,v}\right)^2\right]}{n_i/2 + \lambda_i - 1}$$

until the largest relative change is below $10^{-4}$, with the stopping rule of ComBat. The bracket is the sum of squares of the standardized data of site $i$ around $\gamma^{*}_{i,v}$, written in terms of the sufficient statistics $\hat{\gamma}$ and $\hat{\delta}^2$. The same step therefore applies to every estimator of $\hat{\delta}^2$: the default, `residual_delta` and `nuisance_scale`. It also runs on the aggregator in the decentralized version. The common offset is added back to $\gamma^{*}_{i,v}$, and the result replaces the site rows of $\hat{\gamma}_v$ in the adjustment below, as $\delta^{2*}_{i,v}$ replaces $\hat{\delta}_{i,v}^2$. With `mean_only`, $\delta^{2*}$ enters the posterior of $\gamma$, but the data are not scaled. A reference site keeps its L/S estimates because its data are not changed. Sites with a single subject are left out.

Without nuisance covariates, $\hat{\gamma}$, $\hat{\delta}^2$ and the standardization are those of ComBat. ComCat then equals ComBat with empirical Bayes: it agrees with `combat_family.combat_eb()`, the port of the R package ComBatFamily, to ~1e-13 (`tests/test_comcat_eb.py`).

*Limits.* The prior variance $\bar{\tau}_i^2$ is the variance of the estimates, which includes their sampling variance, and the posterior assumes that $\hat{\gamma}^{c}_{i,v}$ has the variance $\delta^2_{i,v}/n_i$ of a site mean. With nuisance covariates that differ between sites, the site levels at fixed nuisance are estimated less precisely than that, so the location effects are shrunk less than they should be. Some IQMs vary almost only between sites, for example `res_RMS`, a property of the protocol with 2–63 distinct values in the samples below. For such IQMs the site levels at fixed nuisance are not identified. Their L/S estimates then vary widely between features, $\bar{\tau}_i^2$ becomes very large, and the location shrinkage vanishes. The scale shrinkage is not affected.

*Evaluation on real data* (gray matter, 8 mm; ON-Harmony, Buchert, ABIDE, NormSample, ADNI):

- **Split halves.** The subjects of every site were split into random halves. L/S and EB were fitted on one half and compared with L/S on the other, independent half, 10 splits.
  - Site variances: EB reduced the estimated error by 26–87% in all samples, with and without the 7 CAT12 IQMs.
  - Site locations without IQMs: EB reduced the error by 9–72%. With the IQMs, the reduction was 17% (ON-Harmony), 26% (ADNI) and 0% (Buchert, ABIDE, NormSample), as explained above.
- **Held-out traveling subjects, sites only.** Buchert is one subject on 116 scanners; one scan per scanner was held out.
  - Buchert: with L/S, the spread of the held-out scans was 5,364 times that of the raw data, because 1/δ̂ explodes for scanners with 2–3 training scans. With EB it was 0.55 times.
  - ON-Harmony, leaving one subject out: within-subject / total variance was 0.074 raw, 0.112 with L/S and 0.083 with EB.
- **In-sample.** In ADNI with IQMs, the 99th percentile of the harmonized values at a site with two subjects lay 17.5 SD from the voxel mean with L/S, and 1.5 SD with EB.


#### Data adjustment

For each batch $i$, the adjustment is:

$$Y_{v,\text{batch}=i}^{\text{adj}} = \frac{Y_{v,\text{batch}=i}^{\text{std}} - [\mathbf{B}\ \tilde{\mathbf{Z}}]_{\text{batch}=i}\gamma^{*}_v}{\sqrt{\delta^{2*}_{i,v}}}$$

where $\gamma^{*}_v$ is $\hat{\gamma}_v$ with the site rows replaced by their empirical Bayes estimates. With `eb=False` or a single site, $\gamma^{*}_v = \hat{\gamma}_v$ and $\delta^{2*}_{i,v} = \hat{\delta}_{i,v}^2$. The subtraction term removes both the batch-specific additive shift and the smooth nuisance contributions simultaneously, since the full $\gamma^{*}_v$ vector is applied via the combined design matrix $[\mathbf{B}\ \tilde{\mathbf{Z}}]$.


#### Rescaling to original units

The adjusted data are rescaled back to the original data space:

$$Y_{jv}^{\text{ComCat}} = Y_{jv}^{\text{adj}} \cdot \text{RMSE}_v + \bar{\mu}_v + \mathbf{X}_j\hat{\boldsymbol{\beta}}_{X,v}$$

This restores the original data scale while preserving the biological effects encoded in $\mathbf{X}$.


#### Reference site

With a reference site $r$ (`ref_batch`), the data of site $r$ are left unchanged and the other sites are harmonized to it. The grand mean is the mean fitted site and nuisance level of site $r$,

$$\bar{\mu}_v = \frac{1}{N_r} \sum_{j \in r} [\mathbf{B}\ \tilde{\mathbf{Z}}]_j \hat{\boldsymbol{\beta}}_{[\text{batch},Z],v},$$

and the other sites are scaled to the variance of site $r$ instead of the pooled variance:

$$Y_{v,\text{batch}=i}^{\text{adj}} = \frac{Y_{v,\text{batch}=i}^{\text{std}} - [\mathbf{B}\ \tilde{\mathbf{Z}}]_{\text{batch}=i}\gamma^{*}_v}{\sqrt{\delta^{2*}_{i,v} / \hat{\delta}_{r,v}^2}}$$

With empirical Bayes, the location effects of the other sites are shrunk as differences to site $r$. The variance $\hat{\delta}_{r,v}^2$ of site $r$ is not shrunk, because it is the variance of its unchanged data. With `residual_delta` (see *Optional extensions*), the residual variance of every harmonized site then equals that of the reference site, up to the shrinkage of $\delta^{2*}_{i,v}$. With the default $\hat{\delta}$, this holds only if the nuisance terms explain the same share of the within-site variance at both sites.


#### Optional extensions

Three extensions can be enabled in `comcat()` / `comcat_ui()`. All are off by default; the default output is identical to the model described above. Empirical Bayes applies to each of them.

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

The same fit provides a per-feature likelihood-ratio test of nuisance effects on the variance. The deviance difference $\Lambda_v = D_{0,v} - D_{1,v}$ between the models without and with $\mathbf{z}^{*}$ is asymptotically $\chi^2_Q$ under $H_0: \boldsymbol{\theta}_{Z,v} = 0$ only for normal residuals. For standardized residuals $u = e/\sigma$ with kurtosis $\kappa$, $\text{Var}(u^2) = \kappa - 1$, and $\Lambda_v$ is $(\kappa - 1)/2$ times a $\chi^2_Q$ variable, so skewed or heavy-tailed residuals inflate it. It is therefore studentized as in Koenker's (1981) version of the Breusch–Pagan test,

$$\Lambda^{*}_v = \frac{2\Lambda_v}{\hat{\kappa}_v - 1}, \qquad \hat{\kappa}_v - 1 = \frac{1}{N - \text{rank}(\mathbf{W}_0)} \sum_{j=1}^{N} \left(\hat{u}_{0,jv}^2 - 1\right)^2,$$

where $\hat{u}_0$ are the residuals standardized by the variance model under $H_0$, whose design $\mathbf{W}_0$ contains the site and preserved-covariate terms. $\Lambda^{*}_v / Q$ is compared with an $F(Q,\ N - \text{rank}(\mathbf{W}_0))$ distribution, the quasi-likelihood F test (McCullagh and Nelder, 1989). $\Lambda^{*}_v$, its p-value and $\hat{\kappa}_v$ are returned as `scale_lr`, `scale_p` and `scale_kurtosis`. For normal residuals $\hat{\kappa}_v \approx 3$ and $\Lambda^{*}_v \approx \Lambda_v$. The kurtosis is estimated under $H_0$ because residuals standardized by the full model vary less where $\mathbf{z}^{*}$ happens to fit the variance; this made the test liberal in small samples.

In gray matter at 8 mm (2,241 subjects, real residual shapes, three random nuisance variables without effect on the variance), $\Lambda_v$ gave $p < 0.05$ in 12.2% of the voxels (37% of those with |skewness| > 1), and $\Lambda^{*}_v$ in 5.0%. When the mean model has many columns relative to $N$, the residuals contain fewer than $N$ independent values, and the test is liberal for normal residuals as well: in simulations with $N = 80$ and 42 mean-model columns, about 14% of the features had $p < 0.05$ under $H_0$, and $\hat{\boldsymbol{\theta}}_Z$ was attenuated.

The variance model is the one of ComBatLS (Gardner et al., 2025), used for the opposite purpose. ComBatLS models covariate effects on the scale in order to *preserve* them; here, nuisance effects on the scale are modelled in order to *remove* them, while preserved-covariate effects on the scale are kept.

`comcat_from_training()` applies all three extensions to new data with the training estimates (spline knots, $\hat{\boldsymbol{\theta}}_Z$, and the z-scoring of the training data).

## References

Fortin JP, Parker D, Tunç B, Watanabe T, Elliott MA, Ruparel K, et al. (2017). Harmonization of multi-site diffusion tensor imaging data. *NeuroImage*, 161, 149–170.

Gardner M, Shinohara RT, Bethlehem RAI, Romero-Garcia R, Warrier V, Dorfschmidt L, et al. (2025). ComBatLS: A location- and scale-preserving method for multi-site image harmonization. *Human Brain Mapping*, 46(8), e70197.

Johnson WE, Li C, Rabinovic A (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*, 8(1), 118–127.

Koenker R (1981). A note on studentizing a test for heteroscedasticity. *Journal of Econometrics*, 17(1), 107–112.

McCullagh P, Nelder JA (1989). *Generalized Linear Models*, 2nd ed. Chapman and Hall, London.

