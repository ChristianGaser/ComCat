### ComCat: Theoretical framework

ComCat is ComBat (Johnson et al., 2007; Fortin et al., 2017) with continuous nuisance covariates. The site effects are estimated and removed exactly as in ComBat, in location and scale with empirical Bayes; the nuisance effects are penalized smooth functions that are removed in location; the effects of the preserved covariates stay in the data. Without nuisance covariates, ComCat is ComBat.

#### Data model

For feature $v$ ($v = 1, \ldots, P$, e.g., voxels or brain regions) and sample $j$ ($j = 1, \ldots, N$), the model is:

$$Y_{jv} = \mathbf{B}_j\boldsymbol{\beta}_{\text{batch},v} + f_Z(\mathbf{Z}_j)_v + \mathbf{X}_j\boldsymbol{\beta}_{X,v} + \varepsilon_{jv}$$

where $\mathbf{B}$ is the $N \times K$ dummy-coded batch indicator matrix ($K$ = number of sites; reduced to a single intercept column when no site information is provided), $\mathbf{Z}$ is the $N \times Q$ matrix of continuous nuisance covariates (e.g., IQMs), $f_Z$ denotes a smooth function of $\mathbf{Z}$ realized through a penalized B-spline basis expansion (see below), and $\mathbf{X}$ is the $N \times R$ matrix of covariates of interest to be preserved (e.g., age, sex, diagnosis). The batch indicators $\mathbf{B}$ and nuisance covariates $\mathbf{Z}$ together represent the unwanted variation to be removed, while $\mathbf{X}$ contains the biological effects to be preserved.


#### Penalized B-spline expansion of continuous nuisance variables

Each continuous nuisance variable $z_q$ gets one unpenalized linear term (the z-scored variable) and the nonlinear part of a cubic B-spline basis with $d = 10$ functions on equally spaced knots over its observed range:

$$z_q \mapsto \tilde{\mathbf{Z}}_q = [z_q,\ n_1(z_q),\ \ldots,\ n_{d-2}(z_q)]$$

The nonlinear part is the basis minus its least-squares fit by intercept and slope, reparametrized so that the second-order difference penalty of Eilers and Marx (1996) becomes a ridge penalty $\lambda \sum b^2$ on its $d - 2$ coefficients $b$. One smoothing parameter $\lambda$ for all nuisance variables is estimated by REML pooled over all standardized features. This is an empirical Bayes estimate of the size of the nonlinear nuisance effects across features: $\lambda$ is the ratio of the residual variance to the prior variance of the nonlinear coefficients, and a large $\lambda$ gives the linear model. The basis dimension therefore only sets the maximum flexibility; results did not depend on it between 6 and 40 basis functions. `nuisance_model='linear'` uses the linear terms only.

New data are evaluated with the nonlinear part at the boundary of the training range, while the linear part continues. Nuisance variables that do not vary within the sites (or subjects) are not identified and are dropped with a warning; variables with fewer than $d + 2$ distinct values stay linear; missing nuisance values are an error.

The expanded nuisance design matrix is denoted $\tilde{\mathbf{Z}}$.


#### Parameter estimation

All parameters are estimated jointly in a single penalized least squares step on the full design matrix $\mathbf{A} = [\mathbf{B}\ \tilde{\mathbf{Z}}\ \mathbf{X}]$:

$$\hat{\boldsymbol{\beta}} = \left(\mathbf{A}^\top\mathbf{A} + \boldsymbol{\Lambda}\right)^{+}\mathbf{A}^\top Y^\top$$

where $\boldsymbol{\Lambda}$ is diagonal with $\lambda$ for the nonlinear nuisance coefficients and zero otherwise. Because all parameters are estimated jointly, the preserved covariates are accounted for regardless of their correlation with batch or nuisance variables, provided the design matrix is not rank-deficient.


#### Standardization

The grand mean is the sample-average predicted value from the batch and nuisance components:

$$\bar{\mu}_v = \frac{1}{N} \sum_{j=1}^{N} \bigl([\mathbf{B}\ \tilde{\mathbf{Z}}]_j \hat{\boldsymbol{\beta}}_{[\text{batch},Z],v}\bigr)$$

The pooled standard deviation is the root mean squared error of the full-model residuals:

$$\text{RMSE}_v = \sqrt{\frac{1}{N} \sum_{j=1}^{N} \bigl(Y_{jv} - \mathbf{A}_j \hat{\boldsymbol{\beta}}_v\bigr)^2}$$

The standardized data are obtained by removing the grand mean and the preserved covariate effects, then dividing by the pooled standard deviation:

$$Y_v^{\text{std}} = \frac{Y_v - \bar{\mu}_v - \mathbf{X}\hat{\boldsymbol{\beta}}_{X,v}}{\text{RMSE}_v}$$


#### Site and nuisance effects on standardized data

The additive effects of batch and nuisance variables are re-estimated from the standardized data with the reduced design matrix $[\mathbf{B}\ \tilde{\mathbf{Z}}]$ and the same penalty:

$$\hat{\gamma}_v = \left([\mathbf{B}\ \tilde{\mathbf{Z}}]^\top[\mathbf{B}\ \tilde{\mathbf{Z}}] + \boldsymbol{\Lambda}\right)^{+}[\mathbf{B}\ \tilde{\mathbf{Z}}]^\top (Y_v^{\text{std}})^\top$$

The site-specific multiplicative (variance) effects are the within-site variances of the residuals after the additive site and nuisance effects have been removed, as in ComBat, where the site variance is computed after all covariate effects:

$$\hat{\delta}_{i,v}^2 = \text{Var}_i\left(Y_v^{\text{std}} - [\mathbf{B}\ \tilde{\mathbf{Z}}]\hat{\gamma}_v\right)$$

Estimating them from data that still contain the nuisance effects would add the nuisance-explained variance to $\hat{\delta}_{i,v}^2$, and dividing by it would shrink the harmonized residuals by about $1 - R^2$ of the nuisance fit. For the same reason, nothing is scaled with a single site: a variance step would only rescale all residuals. With `mean_only`, $\hat{\delta}_{i,v}^2 = 1$ for all sites.


#### Empirical Bayes estimation of site effects

With at least two sites, the site parameters are estimated with the parametric empirical Bayes method of ComBat (Johnson et al., 2007). With few subjects per site, the L/S estimates $\hat{\gamma}_{i,v}$ and $\hat{\delta}_{i,v}^2$ are noisy, and dividing by a variance estimated from a handful of values can amplify the data of single features many times. Empirical Bayes borrows information across features.

The site rows of $\hat{\gamma}_v$ are the site levels at the zero of the nuisance basis relative to the grand mean. They therefore share a per-feature offset that belongs to the nuisance model and differs between features. Shrinking them directly would leave a site-dependent fraction of this offset in the data. They are first expressed as differences to the size-weighted mean of the sites,

$$\hat{\gamma}^{c}_{i,v} = \hat{\gamma}_{i,v} - \sum_{k} \frac{n_k}{N}\,\hat{\gamma}_{k,v},$$

which are the site effects of ComBat when the nuisance covariates are part of its standardization model. With a reference site $r$, the differences to site $r$ are used. For every site, the effects of all features are assumed to come from common priors,

$$\gamma^{c}_{i,v} \sim N\left(\bar{\gamma}_i, \bar{\tau}_i^2\right), \qquad \delta_{i,v}^2 \sim \text{InvGamma}\left(\lambda_i, \theta_i\right),$$

whose hyperparameters are estimated by the method of moments from $\hat{\gamma}^{c}_{i,v}$ and $\hat{\delta}_{i,v}^2$ over all features. The posterior means are found by iterating

$$\gamma^{*}_{i,v} = \frac{n_i\bar{\tau}_i^2\,\hat{\gamma}^{c}_{i,v} + \delta^{2*}_{i,v}\,\bar{\gamma}_i}{n_i\bar{\tau}_i^2 + \delta^{2*}_{i,v}}, \qquad \delta^{2*}_{i,v} = \frac{\theta_i + \frac{1}{2}\left[(n_i - 1)\,\hat{\delta}_{i,v}^2 + n_i\left(\hat{\gamma}^{c}_{i,v} - \gamma^{*}_{i,v}\right)^2\right]}{n_i/2 + \lambda_i - 1}$$

until the largest relative change is below $10^{-4}$, with the stopping rule of ComBat. The bracket is the sum of squares of the standardized data of site $i$ around $\gamma^{*}_{i,v}$, written in terms of the sufficient statistics $\hat{\gamma}$ and $\hat{\delta}^2$. The common offset is added back to $\gamma^{*}_{i,v}$, and the result replaces the site rows of $\hat{\gamma}_v$ in the adjustment below, as $\delta^{2*}_{i,v}$ replaces $\hat{\delta}_{i,v}^2$. With `mean_only`, $\delta^{2*}$ enters the posterior of $\gamma$, but the data are not scaled. A reference site keeps its L/S estimates because its data are not changed. A site with a single subject gets no site correction: its site effect cannot be separated from that subject's deviation, which the L/S estimate would remove. It gets the size-weighted mean level of the other sites and variance 1, as an unseen site in `comcat_from_training()`.

Without nuisance covariates, $\hat{\gamma}$, $\hat{\delta}^2$ and the standardization are those of ComBat, and ComCat equals ComBat with empirical Bayes: it agrees with `combat_family.combat_eb()`, the port of the R package ComBatFamily, to ~1e-13 (`tests/test_comcat_eb.py`, `tests/test_comcat_vs_legacy.py`).

*Known property of ComBat's site variances, kept on purpose.* The pooled variance of the standardization divides the residual sum of squares by $N$, whereas $\hat{\delta}_{i,v}^2$ divides the sum of squares of site $i$ by $n_i - 1$, and the posterior divides by $n_i$. With few subjects per site, the site variances are therefore too large on average, and the scale step shrinks the harmonized residuals even when the sites do not differ in variance: held out, to about 0.84 of the residual variance of a location-only correction with 3 subjects per site in the fit, 0.92 with 5, 0.97 with 10 and 1.00 with 40 (simulation of the ComCat paper; `tests/test_comcat_scale.py` pins the effect). The shrinkage is the same for ComBat and ComCat, so their comparison is not affected, and effect sizes are not affected either; absolute spreads such as a mean absolute error are. `mean_only=True` avoids it. Earlier versions offered degrees-of-freedom-consistent site variances instead (see *Earlier versions*); they changed no result on real data except with two or three scans per scanner (Buchert), so the site model was kept identical to ComBat's.

*Limits.* The prior variance $\bar{\tau}_i^2$ is the variance of the estimates, which includes their sampling variance, and the posterior assumes that $\hat{\gamma}^{c}_{i,v}$ has the variance $\delta^2_{i,v}/n_i$ of a site mean. With nuisance covariates that differ between sites, the site levels at fixed nuisance are estimated less precisely than that, so the location effects are shrunk less than they should be. Nuisance variables that vary almost only between sites (e.g. `res_RMS`, a property of the protocol) leave the site levels at fixed nuisance unidentified; do not use them.

*Evaluation on real data* (gray matter, 8 mm; ON-Harmony, Buchert, ABIDE, NormSample, ADNI):

- **Split halves.** The subjects of every site were split into random halves. L/S and EB were fitted on one half and compared with L/S on the other, independent half, 10 splits. EB reduced the estimated error of the site variances by 26–87% in all samples and that of the site locations without IQMs by 9–72%.
- **Held-out traveling subject, sites only (Buchert, one scan per scanner held out).** With L/S, the spread of the held-out scans was 5,364 times that of the raw data, because 1/δ̂ explodes for scanners with 2–3 training scans. With EB it was 0.55 times.
- **Held out with IQMs (2026-11).** Without EB, the variance across the held-out Buchert scans was 5,289 times that of the raw data, and the site R² of ADNI rose from 0.079 (raw) to 0.106; with EB they were 0.39 and 0.019. Empirical Bayes is therefore not optional.


#### Data adjustment

For each batch $i$, the adjustment is:

$$Y_{v,\text{batch}=i}^{\text{adj}} = \frac{Y_{v,\text{batch}=i}^{\text{std}} - [\mathbf{B}\ \tilde{\mathbf{Z}}]_{\text{batch}=i}\gamma^{*}_v}{\sqrt{\delta^{2*}_{i,v}}}$$

where $\gamma^{*}_v$ is $\hat{\gamma}_v$ with the site rows replaced by their empirical Bayes estimates. With a single site, $\gamma^{*}_v = \hat{\gamma}_v$ and $\delta^{2*}_{i,v} = 1$. The subtraction term removes both the batch-specific additive shift and the smooth nuisance contributions, since the full $\gamma^{*}_v$ vector is applied via the combined design matrix $[\mathbf{B}\ \tilde{\mathbf{Z}}]$. The nuisance effects are removed in location only: nuisance-dependent residual variance (heteroscedasticity) stays in the data.


#### Rescaling to original units

The adjusted data are rescaled back to the original data space:

$$Y_{jv}^{\text{ComCat}} = Y_{jv}^{\text{adj}} \cdot \text{RMSE}_v + \bar{\mu}_v + \mathbf{X}_j\hat{\boldsymbol{\beta}}_{X,v}$$

This restores the original data scale while preserving the biological effects encoded in $\mathbf{X}$. If the input has no negative values (e.g. tissue densities), negative harmonized values are set to 0 (`clip=False` returns the affine harmonization).


#### Reference site

With a reference site $r$ (`ref_batch`), the data of site $r$ are left unchanged and the other sites are harmonized to it. The grand mean is the mean fitted site and nuisance level of site $r$,

$$\bar{\mu}_v = \frac{1}{N_r} \sum_{j \in r} [\mathbf{B}\ \tilde{\mathbf{Z}}]_j \hat{\boldsymbol{\beta}}_{[\text{batch},Z],v},$$

and the other sites are scaled to the variance of site $r$ instead of the pooled variance:

$$Y_{v,\text{batch}=i}^{\text{adj}} = \frac{Y_{v,\text{batch}=i}^{\text{std}} - [\mathbf{B}\ \tilde{\mathbf{Z}}]_{\text{batch}=i}\gamma^{*}_v}{\sqrt{\delta^{2*}_{i,v} / \hat{\delta}_{r,v}^2}}$$

With empirical Bayes, the location effects of the other sites are shrunk as differences to site $r$. The variance $\hat{\delta}_{r,v}^2$ of site $r$ is not shrunk, because it is the variance of its unchanged data. The residual variance of every harmonized site then equals that of the reference site, up to the shrinkage of $\delta^{2*}_{i,v}$.


#### New data and cross-fitting

All estimates (grand mean, pooled standard deviation, site, nuisance and preserved effects, knots and penalty) can be applied to new scans of the sites seen during estimation (`comcat_from_training()`); scans of other sites get the size-weighted mean site level, so that only their nuisance effects are removed. `comcat_crossfit()` splits the subjects into folds stratified by site and harmonizes every fold with a model estimated on the other folds. In-sample harmonization is optimistic: on real data the mean absolute error of BrainAGE was 6–27% lower in-sample than held out, and with many site terms effect sizes rose by about 10%, for ComBat and ComCat alike. Cross-fit whenever harmonized data are scored by a model trained elsewhere or harmonization methods are compared.


#### Options

**Flexible preserved covariates (`preserve_df`).** Each continuous preserved covariate $x_r$ (at least `preserve_df` + 2 distinct values) is expanded into an unpenalized cubic B-spline basis, $x_r \mapsto \tilde{\mathbf{X}}_r = [b_1(x_r), \ldots, b_d(x_r)]$; binary or categorical covariates stay linear. Estimation is unchanged, with $\tilde{\mathbf{X}}$ in place of $\mathbf{X}$ in the joint model, and the preserved effect $\tilde{\mathbf{X}}_j\hat{\boldsymbol{\beta}}_{\tilde{X},v}$ is added back. With a linear $\mathbf{X}$, nuisance splines can take up the nonlinear part of the effect of $\mathbf{X}$ when $\mathbf{Z}$ depends nonlinearly on $\mathbf{X}$ (for example, image quality is worse in children and older adults). In a simulation with a lifespan-like age curve and seven IQMs that are U-shaped functions of age, the linear age term lost most of the age curve, whereas with `preserve_df` the loss was a few percent (`tests/test_comcat_options.py`). Use it for wide age ranges.

**Effects within subjects (`subject`).** For repeated scans of the same subjects, centred subject indicators can be added to $\mathbf{X}$. Site and nuisance effects are then estimated from the variation within subjects only (location only), so that they cannot take up anatomical differences between subjects. This is the design-appropriate model for traveling-subject and test-retest data with few subjects; with enough subjects (about 70), the cross-sectional model gave the same correction of BrainAGE.


#### Earlier versions

Versions before 2026-11-16 had further options: unpenalized B-splines with a basis dimension `gam_df`, `eb=False`, degrees-of-freedom-consistent site variances with a location-only rule for sites with fewer than four subjects (`combat_compatible=False`, `min_scale_n`), site variances estimated before the nuisance effects were removed, a variance step with one site (`single_site_scale`), and a log-linear model of nuisance-dependent variance (`nuisance_scale`). The code is frozen in `evaluation/comcat_legacy.py` for the evaluations of the ComCat paper, and their theory is in this file at commit 8c3ad69. The current `comcat()` equals the earlier one with `combat_compatible=True, residual_delta=True` bit for bit (`tests/test_comcat_vs_legacy.py`).

## References

Eilers PHC, Marx BD (1996). Flexible smoothing with B-splines and penalties. *Statistical Science*, 11(2), 89–121.

Fortin JP, Parker D, Tunç B, Watanabe T, Elliott MA, Ruparel K, et al. (2017). Harmonization of multi-site diffusion tensor imaging data. *NeuroImage*, 161, 149–170.

Johnson WE, Li C, Rabinovic A (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods. *Biostatistics*, 8(1), 118–127.
