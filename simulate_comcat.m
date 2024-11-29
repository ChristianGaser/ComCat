function [avgD, FPR] = simulate_comcat(a, no_preserving, n, n_sim, n_nuisance, n_perm, mean_only, no_fig)
%[D, FPR] = simulate_comcat(a, no_preserving, n, n_sim, n_nuisance)
% This function simulates data for two samples with a total sample size n 
% n_sim columns.
%
% parameter a (default [1.0 0.5 0.5 1.0]):
%  a(1) - Amplitude of EoI.
%  a(2) - Amplitude of nuisance effect.
%  a(3) - Amplitude of multiplicative effect.
%  a(4) - covariance between nuisance and covariate of interest
%
%  no_preserving - Controls whether we should preserve our EoI or not
%  n             - Total sample size (rows of Y, must be even)
%  n_sim         - Number of simulations (columns of Y)
%  n_nuisance    - Number of nuisance parameters
%
% The distribution of the effect size D is plotted for the adjusted data (with AnCova
% in GLM) and for the harmonized data. In addition, the difference of the
% effect size D between harmonized and adjusted data is plotted to evaluate
% the increase in sensitivity.
% Although the effect size is independent of the degrees of freedom, the empty 
% columns for nuisance in the design matrix of the GLM for the
% harmonized data are filled with (orthogonal) noise to balance the degrees of
% freedom.
%
% Please note that the EoI and the nuisance are the same for all columns of Y. 
% However, Y is created by adding different columns with noise.

% Amplitude settings for effects of interest and nuisance variables
a1 = 1.0;  % Amplitude of EoI.
a2 = 0.2; % Amplitude of nuisance effect.
a3 = 0.0;  % Amplitude of multiplicative effect.
a4 = 0.5;  % covariance between nuisance and covariate of interest

% some defaults
show_T = 0; % Controls whether to show T distribution or effect size D.
apply_2step_correction = 1; % Correction by Zhao
n_simu = 2; % skip 3rd simulation

if nargin < 1
  a = [a1 a2 a3 a4];
end

if nargin < 2
  no_preserving = 0; % Controls whether we should preserve our EoI or not
end

% Dataset configuration: total size, number of simulations, and nuisance parameters
if nargin < 3
  n = 1000; % Total sample size (must be even).
end
if nargin < 4
  n_sim = 500; % Number of simulations.
end
if nargin < 5
  n_nuisance = 1; % Number of nuisance parameters.
end
if nargin < 6
  n_perm = 0; % number of permutations
end
if nargin < 7
  mean_only = 0; % mean only approach
end
if nargin < 8
  no_fig = 0; % suppress figures
end

% Initialize random number generator for reproducibility
rng('shuffle');
%rng(0);

% create initial X (covariate of interest) and Z (nuisance)
X0 = randn(n, 1);
noise_ortho = spm_orth([X0 randn(n,n_nuisance)]); % make orthogonal to X
noise_ortho = noise_ortho(:,2:n_nuisance+1); % Orthogonal.
D0 = [X0 noise_ortho];

% covariance structure between nuisance and covariate of interest
A = [1.0, a(4); a(4), 1.0]; 

% use Cholesky decomposition to get a symmetric positive definite matrix
R = chol(A);

Z = [];
% go through all nuisance parameters
for i = 1:n_nuisance
  D = D0(:,[1 i+1])*R;
  Z = [Z D(:,2)];
end
X = D0(:,1);

% Generate noise for simulations
E = randn(n,n_sim);

% Initialize variables for storing statistical analysis results
T    = cell(n_simu,1); % t-values for observed data.
D    = cell(n_simu,1); % Effect size (Cohen's D) for observed data.

FPR  = zeros(n_simu,1); % False positive rate.
avgD = zeros(n_simu,1); % Average effect suze.

%-------------------------------------------------------------------------------
% Create the simulated signal with noise and nuisance effects
%-------------------------------------------------------------------------------
% only add noise E and multiplicative once even if nuisance has several entries
% because otherwise noise will be added for each step...
Y = a(1)*X + exp(a(3)*Z(:,1)).*E; % Base signal with multiplicative effect (of 1st nuisance parameter)
Y0 = Y';
for i=1:n_nuisance
  Y = Y + a(2)*Z(:,i); % Additive nuisance effect.
end
Y = Y';

%-------------------------------------------------------------------------------
% Harmonize data using ComCat
%-------------------------------------------------------------------------------
nuisance_polynomial = 1; % Determines if a polynomial model is used for nuisance.
if no_preserving
  [Y_comcat] = comcat(Y, [], Z, [], mean_only, nuisance_polynomial, 0);
else
  [Y_comcat] = comcat(Y, [], Z, X, mean_only, nuisance_polynomial, 0);
end

%-------------------------------------------------------------------------------
% Residualized Adjustment: Adjust data by fit of nuisance parameters only
%-------------------------------------------------------------------------------
if no_preserving
  Beta  = Y*pinv([Z, ones(n,1)]'); % Regression coefficients.
  Y_adjusted = Y - Beta(:,1:n_nuisance)*Z'; % Adjust data by removing nuisance effect.

%-------------------------------------------------------------------------------
% Adjust data considering nuisance parameters (Ancova)
%-------------------------------------------------------------------------------
else
  XZ = [Z, X, ones(n,1)]; % Design matrix.
  Beta  = Y*pinv(XZ'); % Regression coefficients.
  Y_adjusted = Y - Beta(:,1:n_nuisance)*Z'; % Adjust data by removing nuisance effect.
end

%-------------------------------------------------------------------------------
% GLM AnCova
%-------------------------------------------------------------------------------
c = [zeros(1,n_nuisance) 1 0]';
XZ = [Z, X, ones(n,1)];

if ~no_fig
  figure(29)
  imagesc(XZ)
  title('Design Matrix XZ')
end

[T{1}, trRV, beta_hat] = calc_GLM(Y,XZ,c);
D{1} = double(2*T{1}/sqrt(trRV));
Thresh = spm_invTcdf(1-(0.05),trRV);
FPR(1) = sum(T{1}>Thresh)/n_sim;
if a(1) == 0; fprintf('\nRejection rate FP = %3.4f for AnCova\n',FPR(1)); end

ind_c = 2;
beta_hat_ancova  = beta_hat(:,ind_c);

if n_perm
  Thresh0 = spm_invTcdf(1-(0.05/n_sim),trRV);
  
  % GLM AnCova (null distribution)
  [T0{1}, FPR2(1)] = calc_GLM_null(Y,XZ,c,n_perm,Thresh0);
  fprintf('\n\nRejection rate FP = %3.4f (FWE) for AnCova (%d permutations)\n',FPR2(1),n_perm);
  fprintf('Rejection rate FP = %3.4f for AnCova (%d permutations)\n\n',sum(T0{1}>Thresh)/n_sim/n_perm,n_perm);
end


%-------------------------------------------------------------------------------
% GLM for ComCat harmonized data
%-------------------------------------------------------------------------------
% add different noise estimations as dummy parameter in GLM
% to equalize df between original data and adjusted data
%c = [zeros(1,n_nuisance) 1 0]';
%XZ = [randn(size(Z)), X, ones(n,1)];

c = [1 0]';
XZ = [X, ones(n,1)];
r{2} = rank(XZ);

if apply_2step_correction
  Id = eye(n);
  h1 = X * pinv(X' * X) * X';
  Z2 = [Z, ones(n,1)];
  h12 = Z2 * pinv(Z2' * (Id - h1) * Z2) * Z2' * (Id - h1);
  reduce = Id - h12;
  M = reduce * reduce';
  [v, d] = eig(M);
  d = diag(d);
  noise = 1 / n;
  err = sum(d) * noise;
  d(d < 0.0001) = err;
  k = v * diag(d) * v';
  s = chol(inv(k))';
  
  Y_comcat = Y_comcat*s;
  %XZ = s'*XZ;
end

[T{2}, trRV, beta_hat] = calc_GLM(Y_comcat,XZ,c);
D{2} = double(2*T{2}/sqrt(trRV));
Thresh = spm_invTcdf(1-(0.05),trRV);
FPR(2) = sum(T{2}>Thresh)/n_sim;
if a(1) == 0; fprintf('Rejection rate FP = %3.4f for ComCat harmonized data\n',FPR(2)); end

ind_c = 1;
beta_hat_comcat  = beta_hat(:,ind_c);

if n_perm
  Thresh0 = spm_invTcdf(1-(0.05/n_sim),trRV);
  
  % GLM AnCova (null distribution)
  [T0{2}, FPR2(2)] = calc_GLM_null(Y,XZ,c,n_perm,Thresh0);
  fprintf('\n\nRejection rate FP = %3.4f (FWE) for ComCat harmonized data (%d permutations)\n',FPR2(2),n_perm);
  fprintf('Rejection rate FP = %3.4f for ComCat harmonized data (%d permutations)\n\n',sum(T0{2}>Thresh)/n_sim/n_perm,n_perm);
end
%-------------------------------------------------------------------------------
% Use LME
%-------------------------------------------------------------------------------

if n_simu > 2
  if 0
  
    str_model = 'LME';
    c = [zeros(1,n_nuisance) 1 0]'; % Contrast vector for GLM.
    
    Ti = zeros(n_sim,1);
    beta_hat_lme = zeros(n_sim,1);
    for i=1:n_sim
      Yi = Y(i,:)';
      lmm = fitlme(table(Z, X, Yi),'Yi ~ 1 + X + (Z|X)');
      beta_hat_lme(i,:) = lmm.Coefficients.Estimate(2);
      Ti(i,:) = lmm.Coefficients.tStat(2);
    end
    
    T{3} = Ti;
    D{3} = double(2*T{3}/sqrt(trRV));
    Thresh = spm_invTcdf(1-(0.05),trRV);
    FPR(3) = sum(T{3}>Thresh)/n_sim;
    if a(1) == 0; fprintf('Rejection rate FP = %3.4f for LME\n',FPR(3)); end
  
  %-------------------------------------------------------------------------------
  % GLM for adjusted data
  %-------------------------------------------------------------------------------
  % add different noise estimations as dummy parameter in GLM
  % to equalize df between original data and adjusted data
  %c = [zeros(1,n_nuisance) 1 0]';
  %XZ = [randn(size(Z)), X, ones(n,1)];
  
  else
  
    str_model = 'GLM Adjusted';
    c = [1 0]';
    XZ = [X, ones(n,1)];
    
    [T{3}, trRV, beta_hat] = calc_GLM(Y_adjusted,XZ,c);
    D{3} = double(2*T{3}/sqrt(trRV));
    Thresh = spm_invTcdf(1-(0.05),trRV);
    FPR(3) = sum(T{3}>Thresh)/n_sim;
    if a(1) == 0; fprintf('Rejection rate FP = %3.4f for adjusted data\n',FPR(3)); end
    
    ind_c = 1;
    beta_hat_lme  = beta_hat(:,ind_c);
    
    if n_perm
      Thresh0 = spm_invTcdf(1-(0.05/n_sim),trRV);
      
      % GLM AnCova (null distribution)
      [T0{3}, FPR2(3)] = calc_GLM_null(Y,XZ,c,n_perm,Thresh0);
      fprintf('\n\nRejection rate FP = %3.4f (FWE) for adjusted data (%d permutations)\n',FPR2(3),n_perm);
      fprintf('Rejection rate FP = %3.4f for adjusted data (%d permutations)\n\n',sum(T0{3}>Thresh)/n_sim/n_perm,n_perm);
    end
    
    % give warning if ranks of design matrices differ
    if r{1} ~= r{2}
      fprintf('WARNING: Design matrices have different ranks: %d %d\n',r{1},r{2});
    end
  end
end

%-------------------------------------------------------------------------------
% Print correlations
%-------------------------------------------------------------------------------
name = sprintf('\nReal effects (scaling=%g) / (scal_add=%g/scal_mult=%g)\n',a(1),a(2),a(3));
fprintf(name);

cc = corrcoef([X,Z]);
fprintf('Correlation of nuisance variable with preserving variable (EoI): %5.3f\n',mean(abs(cc(1,2:end))));

if n_simu > 2
  str_data    = {'AnCova (GLM)', 'GLM ComCat harmonized data ', str_model};
else
  str_data    = {'AnCova (GLM)', 'GLM ComCat harmonized data '};
end

% print mean correlations between nuisance and data
if n_sim < 5000 % take too long otherwise
  cc0 = corrcoef([Z Y']);
  cc1 = corrcoef([Z Y_comcat']);
  if n_simu > 2
    cc2 = corrcoef([Z Y_adjusted']);
    fprintf('Mean correlation between nuisance and Y:\n%s\t%g\n%s\t%g\n%s\t%g\n',str_data{1},mean(cc0(1,2:end)),str_data{2},mean(cc1(1,2:end)),...
     str_data{3},mean(cc2(1,2:end)));
  else
    fprintf('Mean correlation between nuisance and Y:\n%s\t%g\n%s\t%g\n',str_data{1},mean(cc0(1,2:end)),str_data{2},mean(cc1(1,2:end)));
  end
end
if show_T
  D = T;
  str_T = 'T';
else
  str_T = 'Effect size D';
end

fprintf('\n');
for i=1:n_simu
  avgD(i) = mean(D{i});
  fprintf('Mean of %s  = %3.5f %s\n',str_T,avgD(i),str_data{i});
end

%-------------------------------------------------------------------------------
% Plot distributions
%-------------------------------------------------------------------------------
if no_fig
  return;
  if ~nargout
    clear avgD FPR
  end
end

if exist('cat_io_colormaps')
  col = cat_io_colormaps('nejm',4);
else
  col = [0.7373 0.2353 0.1608; 0 0.4471 0.7098; 0.8824 0.5294 0.1529; 0.1255 0.5216 0.3059];
end

figure(9)
if n_simu > 2
  hist([beta_hat_ancova, beta_hat_comcat, beta_hat_lme],25);
  legend({'Beta AnCova','Beta ComCat','Beta Adjusted'});
else
  hist([beta_hat_ancova, beta_hat_comcat],25);
  legend({'Beta AnCova','Beta ComCat'});
end
set(gca,'Colormap',col)

figure(10)
subplot(4,1,1); plot(Y0(:,1)); ylim([-5 5]); title('Signal with added noise')
subplot(4,1,2); plot(Y(:,1)); ylim([-5 5]);          title('Signal with nuisance effects and added noise')
subplot(4,1,3); plot(a(2)*Z); ylim([-2 2]);  title('Nuisance effects')
subplot(4,1,4); plot(Y_comcat(:,1)); ylim([-5 5]);   title('Harmonized signal')
mn = 1e9; mx = -1e9;

for i=1:numel(D)
  minD = min(D{i});
  maxD = max(D{i});
  if minD < mn, mn = minD; end
  if maxD > mx, mx = maxD; end
end

x  = linspace(0.9*mn,mx,100);

figure(11)
for i=1:n_simu
  y = D{i};
  if exist('fitdist')
    ny = numel(y);
    H1 = hist(y,x);
    pd = fitdist(y,'kernel');

    [bincounts,binedges] = histcounts(y,x);

    % Normalize the density to match the total area of the histogram
    Hfit(i,:) = ny * (binedges(2)-binedges(1)) * pdf(pd,x);

    H(i,:) = H1;
  else
    Hfit(i,:) = hist(y,x);
  end
  X1(i,:) = x;
  
end  
  
HP  = plot(X1', Hfit');
hold on
if exist('fitdist')
  HP1 = plot(X1', H');
end
hold off

if exist('cat_io_colormaps')
  col = cat_io_colormaps('nejm',length(HP));
else
  col = [0.7373 0.2353 0.1608; 0 0.4471 0.7098; 0.8824 0.5294 0.1529; 0.1255 0.5216 0.3059];
end

for i = 1:length(HP)
  set(HP(i),'LineWidth',2,'Color',col(i,:));
  if exist('fitdist')
    set(HP1(i),'LineWidth',1,'Linestyle',':','Color',col(i,:));
  end  
end

legend(str_data(1:n_simu));
title({str_T,name})

if ~nargout
  clear avgD FPR
end

%-------------------------------------------------------------------------------
function [T, FP] = calc_GLM_null(Y,X,c,n_perm,Thresh)
% compute null distribution using permutation
% nuisance parameters are considered using Freedman-Lane method

% always use same permuation scheme
rng(0);

T = [];
FP = 0;

n = size(Y,2);

% find EoI entries
[indi, ~] = find(c~=0);
ind_X = unique(indi)';

% 0 - Draper-Stoneman
% 1 - Freedman-Lane
% 2 - Smith
nuisance_method = 2;

% Guttman partioning of design matrix into effects of interest X and nuisance variables Z
if nuisance_method > 0
  Z = X(:,~ind_X);
    
  Hz = Z*pinv(Z);
  Rz = eye(n) - Hz;
end

for i=1:n_perm

  Pset = sparse(n,n);
  r = randperm(n);
  for i=1:n
    Pset(r(i),i) = 1;
  end

  Xperm = X;
  
  switch nuisance_method 
  case 0 % Draper-Stoneman is permuting X
    Xperm(:,ind_X) = Pset*Xperm(:,ind_X);
  case 1 % Freedman-Lane is permuting Y
    Xperm = X;
  case 2 % Smith method is additionally orthogonalizing X with respect to Z
    Xperm(:,ind_X) = Pset*Rz*Xperm(:,ind_X);
  end
  
  % Freedman-Lane permutation of data
  if nuisance_method == 1
    T0 = calc_GLM(Y*(Pset'*Rz),Xperm,c);
  else
    T0 = calc_GLM(Y,Xperm,c);
  end

  T = [T; T0];
  if max(T0) > Thresh
    FP = FP + 1;
  end
end
FP = FP/n_perm;

%-------------------------------------------------------------------------------
function [T, trRV, Beta] = calc_GLM(Y,X,c)
% compute T-statistic using GLM
%
% Y        - masked data as vector
% xX       - design structure
% xCon     - contrast structure
% ind_mask - index of mask image
% dim      - image dimension
%
% Output:
% T        - T/F-values
% trRV     - df

pKX = pinv(X);

n_data = size(X,1);

Beta = Y*pKX';
res0 = Beta*(single(X'));
res0 = res0 - Y; %-Residuals
res0 = res0.^2;
ResSS = double(sum(res0,2));
clear res0

trRV = n_data - rank(X);
ResMS = ResSS/trRV;

Bcov = pKX*pKX';
con = Beta*c;

T = con./(eps+sqrt(ResMS*(c'*Bcov*c)));
