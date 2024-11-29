function [Y_harmonized, gamma, delta] = combat(Y, batch, nuisance, preserve, parametric, mean_only, poly_degree, eb, verbose)
%-------------------------------------------------------------------------------
%{
MIT License

Copyright (c) 2020 Jean-Philippe Fortin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
%}

% This is a heavily modified version of combat.m and its subfunctions
% https://github.com/Jfortin1/ComBatHarmonization/tree/master/Matlab/scripts
  if nargin < 9, verbose = 0; end
  if nargin < 8, eb = 0; end
  if nargin < 7, poly_degree = 2; end
  if nargin < 6, mean_only = 0; end
  if nargin < 5, parametric = 1; end
  if nargin < 4, error('Syntax: Y_harmonized = combat(Y, batch, nuisance, preserve, parametric, mean_only, eb, poly_degree, verbose);'); end
  
  % transpose if necessary
  [m,n] = size(batch);    if m<n, batch = batch'; end
  [m,n] = size(nuisance); if m<n, nuisance = nuisance'; end
  [m,n] = size(preserve); if m<n, preserve = preserve'; end
  
  % get array size for any non-empty parameter
  if ~isempty(batch)
    n_array = size(batch,1);
  elseif ~isempty(nuisance)
    n_array = size(nuisance,1);
  elseif ~isempty(preserve)
    n_array = size(preserve,1);
  end
  
  n_nuisance = size(nuisance,2);
  n_preserve = size(preserve,2);

  % disable EB if nuisance harmonization is defined
  if n_nuisance && eb
    fprintf('WARNING: Empirical Bayes is disabled because it cannot be combined with nuisance harmonization\n');
    eb = 0;
  end
  
  % transpose Y if necessary
  if size(Y,2) ~= n_array
    Y = Y';
    transp = 1;
  else
    transp = 0;
  end

  if size(Y,1) == 1 && eb
    fprintf('WARNING: Empirical Bayes can only be used if Y is a matrix. EB-option was disabled.\n');
    eb = 0;
  end
  
  % use only those data entries where std is >0 and finite to skip background
  % and masked areas
  dim = size(Y);
  sd0 = std(Y,[],2);
  avg = mean(Y,2);

  % we check the distribution of the values
  ind = sd0 > 0 & isfinite(sd0);

  Y = Y(ind,:);
  ind_NaN = isnan(sd0);
  
  % apply polynomial expansion and orthogonalization to nuisance variables
  if n_nuisance
    if verbose, fprintf('[combat] Polynomial extension with degree %d\n', poly_degree); end
    for i = 1:n_nuisance
      tmp_nuisance{i} = cat_stat_polynomial(nuisance(:,i),poly_degree);
    end
    nuisance = cell2mat(tmp_nuisance);
    n_nuisance = size(nuisance,2);
  end
  
  % we need an intercept if batch is not defined
  if isempty(batch)
    if isempty(nuisance)
      Y_harmonized = Y;
      return
    end
    batch = ones(n_array,1);
  end
  
  levels = unique(batch);
  batchmod = categorical(batch);
  if ~isempty(batchmod)
    batchmod = dummyvar({batchmod}); % transform batch/site coding to 0/1 columns for GLM
    n_batch = size(batchmod,2);
    if verbose && n_batch > 1, fprintf('[combat] Found %d different sites\n', n_batch); end
  else
    n_batch = 0;
  end
  batchmod = double(batchmod);
  
  batches = cell(0);
  for i=1:n_batch
    batches{i}=find(batch == levels(i));
  end
  n_batches = cellfun(@length,batches);

  % Creating design matrix and ignoring intercept:
  ind_batch    = 1:n_batch;
  ind_nuisance = n_batch+1:n_batch+n_nuisance;
  ind_preserve = n_batch+n_nuisance+1:n_batch+n_nuisance+n_preserve;

  % Creating design matrix 
  X = [batchmod nuisance preserve];

  if ~isempty(preserve) && verbose
    fprintf('[combat] Preserving %d covariate(s) \n',numel(ind_preserve));
  end
  
  % Check if the design is confounded
  if rank(X)<size(X,2)
    nn = size(X,2);
    if nn==(n_batch+ind_nuisance+1) 
      error('Error. The covariate is confounded with batch. Remove the covariate and rerun ComBat.')
    end
    if nn>(n_batch+ind_nuisance+1)
      temp = X(:,(n_batch+ind_nuisance+1):nn);
      if rank(temp) < size(temp,2)
        error('Error. The covariates are confounded. Please remove one or more of the covariates so the design is not confounded.')
      else 
        error('Error. At least one covariate is confounded with batch. Please remove confounded covariates and rerun ComBat.')
      end
    end
  end

  if verbose, fprintf('[combat] Standardizing Data across features\n'); end
  B_hat = pinv(X)*Y';
    
  % Standardization Model
  grand_mean = (n_batches/n_array)*B_hat(ind_batch,:);
%  grand_mean = mean(X(:,[ind_batch ind_nuisance])*B_hat([ind_batch],:)); % orig
grand_mean(:)=0;
  std_pooled = sqrt(mean((Y-(X*B_hat)').^2,2));
  stand_mean = grand_mean'*ones(1,n_array);
  
  % Making sure pooled std are not zero:
  wh = find(std_pooled==0);
  std_pooled_notzero = std_pooled;
  std_pooled_notzero(wh) = [];
  std_pooled(wh) = median(std_pooled_notzero);
std_pooled(:)=1;
  stand_mean = stand_mean + (X(:,ind_preserve)*B_hat(ind_preserve,:))';
  s_data0 = (Y-stand_mean)./(std_pooled*ones(1,n_array));
  clear Y
  
  % Get regression batch effect parameters
  if verbose, fprintf('[combat] Fitting L/S model and finding priors\n'); end
  X_nuisance = X(:,[ind_batch ind_nuisance]); % orig
  gamma_hat = pinv(X_nuisance)*s_data0';
  
  delta_hat = [];
  for i=1:n_batch
    indices = batches{i};
    if mean_only
      delta_hat = [delta_hat; ones(size(s_data0,1))];
    else
      delta_hat = [delta_hat; var(s_data0(:,indices)')];
    end
  end
  for i=n_batch+1:n_batch+n_nuisance
    if mean_only
      delta_hat = [delta_hat; ones(n_nuisance,dim(1))];
    else
      delta_hat = [delta_hat; var(s_data0')];
    end
  end
  
  % Find parametric priors:
  gamma_bar = mean(gamma_hat');
  t2 = var(gamma_hat');
  
  delta_hat_cell = num2cell(delta_hat,2);
  a_prior=[]; b_prior=[];

  if ~mean_only && eb
    for i=1:n_batch
      a_prior=[a_prior aprior(delta_hat_cell{i})];
      b_prior=[b_prior bprior(delta_hat_cell{i})];
    end
    for i=n_batch+1:n_batch+n_nuisance
      a_prior=[a_prior aprior(delta_hat_cell{i})];
      b_prior=[b_prior bprior(delta_hat_cell{i})];
    end
  end
  
  if eb
    if parametric
      if verbose, fprintf('[combat] Finding parametric adjustments\n'); end
      gamma_star =[]; delta_star=[];
      for i=1:n_batch
        indices = batches{i};
        if mean_only
          gamma_star = [gamma_star; postmean(gamma_hat(i,:),gamma_bar(i),1,1,t2(i))];
          delta_star = [delta_star; ones(size(s_data0,1))];
        else
          temp = itSol(s_data0(:,indices),gamma_hat(i,:),delta_hat(i,:),gamma_bar(i),t2(i),a_prior(i),b_prior(i), 0.001);
          gamma_star = [gamma_star; temp(1,:)];
          delta_star = [delta_star; temp(2,:)];
        end
      end
      % for nuisance variables we cannot use EB approach and use
      % gamma_hat/delta_hat
      for i=n_batch+1:n_batch+n_nuisance
        if mean_only
          gamma_star = [gamma_star; gamma_hat(i,:)];
          delta_star = [delta_star; ones(size(s_data0,1))];
        else
          gamma_star = [gamma_star; gamma_hat(i,:)];
          delta_star = [delta_star; delta_hat(i,:)];
        end
      end
    else
      gamma_star =[]; delta_star=[];
      if verbose, fprintf('[combat] Finding non-parametric adjustments\n'); end
      for i=1:n_batch
        indices = batches{i};
        if mean_only
          delta_hat(i,:) = ones(size(s_data0,1));
        end
        temp = inteprior(s_data0(:,indices),gamma_hat(i,:),delta_hat(i,:));
        gamma_star = [gamma_star; temp(1,:)];
        delta_star = [delta_star; temp(2,:)];
      end
      % for nuisance variables we cannot use EB approach and use
      % gamma_hat/delta_hat
      for i=n_batch+1:n_batch+n_nuisance
        if mean_only
          gamma_star = [gamma_star; gamma_hat(i,:)];
          delta_star = [delta_star; ones(size(s_data0,1))];
        else
          gamma_star = [gamma_star; gamma_hat(i,:)];
          delta_star = [delta_star; delta_hat(i,:)];
        end
      end
    end
  else
    gamma_star = gamma_hat;
    delta_star = delta_hat;
  end
  
  if verbose > 1
    fprintf('Gamma %g\n',gamma_star);
    fprintf('Delta %g\n',delta_star);
  end
  if verbose, fprintf('[combat] Adjusting the Data\n'); end
  
  Y_harmonized0 = s_data0; clear s_data0;
  for i=1:n_batch
    indices = batches{i};
    denom = sqrt(delta_star(i,:))'*ones(1,n_batches(i));
    numer = (Y_harmonized0(:,indices)-(X_nuisance(indices,:)*gamma_star)');
    Y_harmonized0(:,indices) = numer./denom;
  end

  % only run this if no batch was defined
  if ~n_batch && n_nuisance
    denom = sqrt(delta_star)'*ones(1,n_array);
    numer = (Y_harmonized0-(X_nuisance*gamma_star)');
    Y_harmonized0 = numer./denom;
  end
  
  % return indexed data and rescue NaNs for ComBat
  Y_harmonized = zeros(dim);
  Y_harmonized(ind,:) = (Y_harmonized0.*(std_pooled*ones(1,n_array)))+stand_mean;
  Y_harmonized(ind_NaN,:) = NaN;
  
  if nargout > 1
    ng = size(gamma_hat,1);
    gamma = zeros(ng,dim(1));
    for i=1:ng
      gamma(i,ind) = gamma_star(i,:); 
    end
  end
  
  if nargout > 2
    nd = size(delta_hat,1);
    delta = zeros(nd,dim(1));
    for i=1:nd
      delta(i,ind) = delta_star(i,:); 
    end
  end
  
  % transpose result if Y was transposed
  if transp, Y_harmonized = Y_harmonized'; end
end

%-------------------------------------------------------------------------------
function y = aprior(gamma_hat)
% This is a modified version of aprior.m
% https://github.com/Jfortin1/ComBatHarmonization/tree/master/Matlab/scripts
  m = mean(gamma_hat);
  s2 = var(gamma_hat);
  y=(2*s2+m^2)/s2;
end

%-------------------------------------------------------------------------------
function y = bprior(gamma_hat)
% This is a  modified version of bprior
% https://github.com/Jfortin1/ComBatHarmonization/tree/master/Matlab/scripts
  m = mean(gamma_hat);
  s2 = var(gamma_hat);
  y=(m*s2+m^3)/s2;
end

%-------------------------------------------------------------------------------
function adjust = itSol(sdat,g_hat,d_hat,g_bar,t2,a,b, conv)
% This is a heavily modified version of itSol
% https://github.com/Jfortin1/ComBatHarmonization/tree/master/Matlab/scripts

  global debug
  
  g_old = g_hat;
  d_old = d_hat;
  change = 1;
  count = 0;
  n = size(sdat,2);
  while change>conv 
    g_new = postmean(g_hat,g_bar,n,d_old,t2);
    sum2  = sum(((sdat-g_new'*ones(1,size(sdat,2))).^2)');
    d_new = postvar(sum2,n,a,b);

    g_change = max(abs(g_new-g_old)./g_old);
    d_change = max(abs(d_new-d_old)./d_old);
    change   = max(d_change, g_change);
    
    g_old = g_new;
    d_old = d_new;
    count = count+1;

    if debug
      fprintf('Iteration %03d: d-change %g\tg-change %g\n',count, d_change, g_change);
    end
    
    if count > 1000
      warning('EB algorithm did not converge in 1000 steps.');
      break;
    end
  end
  adjust = [g_new; d_new];
end

%-------------------------------------------------------------------------------
% This is a  modified version of postmean
% https://github.com/Jfortin1/ComBatHarmonization/tree/master/Matlab/scripts
function y = postmean(g_hat ,g_bar, n,d_star, t2)
  y=(t2*n.*g_hat+d_star.*g_bar)./(t2*n+d_star);
end

%-------------------------------------------------------------------------------
% This is a  modified version of postvar
% https://github.com/Jfortin1/ComBatHarmonization/tree/master/Matlab/scripts
function y = postvar(sum2,n,a,b)
  y=(.5.*sum2+b)./(n/2+a-1);
end
