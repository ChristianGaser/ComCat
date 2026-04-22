function [Y_harmonized, beta_hat, gamma_hat, delta_hat] = comcat(Y, batch, Z, X, mean_only, poly_degree, verbose)
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
  if nargin < 7, verbose = 0; end
  if nargin < 6, poly_degree = 3; end
  if nargin < 5, mean_only = 0; end
  if nargin < 4, error('Syntax: Y_harmonized = comcat(Y, batch, nuisance, preserve, mean_only, poly_degree, verbose);'); end
  
  % transpose if necessary
  [m,n] = size(batch); if m<n, batch = batch'; end
  [m,n] = size(Z); if m<n, Z = Z'; end
  [m,n] = size(X); if m<n, X = X'; end
  
  % get array size for any non-empty parameter
  if ~isempty(batch)
    n_array = size(batch,1);
  elseif ~isempty(Z)
    n_array = size(Z,1);
  elseif ~isempty(X)
    n_array = size(X,1);
  end

  % we need an intercept if batch is not defined and have to enable mean_only
  if isempty(batch)
    if isempty(Z)
      Y_harmonized = Y;
      return
    end
    batch = ones(n_array,1);
    mean_only = 1;
  end
    
  n_Z = size(Z,2);
  n_X = size(X,2);
  
  % transpose Y if necessary
  if size(Y,2) ~= n_array
    Y = Y';
    transp = 1;
  else
    transp = 0;
  end

  % we have to save memory
  Y = single(Y);
  
  % use only those data entries where std is > 0 and finite to skip background
  % and masked areas
  dim = size(Y);
  sd0 = std(Y,[],2);
  avg = mean(Y,2);
  
  % we check that SD exceeds 0
  ind_mask = sd0 > 0 & isfinite(sd0);
  
  Y = Y(ind_mask,:);
  ind_NaN = isnan(sd0);
  
  % apply polynomial expansion and orthogonalization to nuisance variables
  if n_Z
    if verbose && poly_degree > 1, fprintf('[ComCAT] Polynomial extension of nuisance parameter(s) with degree %d\n', poly_degree); end
    tmp_nuisance = cell(1,n_Z);
    for i = 1:n_Z
      tmp_nuisance{i} = polynomial(Z(:,i),poly_degree);
    end
    Z = cell2mat(tmp_nuisance);
    n_Z = size(Z,2);
  end
  
  levels = unique(batch);
  batchmod = categorical(batch);
  if ~isempty(batchmod)
    batchmod = dummyvar({batchmod}); % transform batch/site coding to 0/1 columns for GLM
    n_batch = size(batchmod,2);
    if verbose && n_batch > 1, fprintf('[ComCAT] Found %d different sites\n', n_batch); end
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
  ind_nuisance = n_batch+1:n_batch+n_Z;
  ind_preserve = n_batch+n_Z+1:n_batch+n_Z+n_X;

  % Creating design matrix 
  XZ = [batchmod Z X];
  if ~isempty(X) && verbose
    fprintf('[ComCAT] Preserving %d covariate(s) \n',numel(ind_preserve));
  end
  
  % Check if the design is confounded
  if rank(XZ)<size(XZ,2)
    nn = size(XZ,2);
    if nn==(n_batch+n_Z+1)
      error('Error. The covariate is confounded with batch. Remove the covariate and rerun ComCAT.')
    end
    if nn>(n_batch+n_Z+1)
      temp = XZ(:,(n_batch+n_Z+1):nn);      
      if rank(temp) < size(temp,2)
        warning('Error. The covariates are confounded. Please remove one or more of the covariates so the design is not confounded.')
      else 
        error('Error. At least one covariate is confounded with batch. Please remove confounded covariates and rerun ComCAT.')
      end
    end
  end

  if verbose, fprintf('[ComCAT] Standardizing Data across features\n'); end
  beta_hat = pinv(XZ)*Y';
    
  % Standardization Model
  grand_mean = mean(XZ(:,[ind_batch ind_nuisance])*beta_hat([ind_batch ind_nuisance],:));
  std_pooled = sqrt(mean((Y-(XZ*beta_hat)').^2,2));
  
  % Making sure pooled std are not zero
  wh = find(std_pooled==0);
  std_pooled_notzero = std_pooled;
  std_pooled_notzero(wh) = [];
  std_pooled(wh) = median(std_pooled_notzero);

  % save memory in correcting grand_mean per subject and update Y
  for i=1:n_array
    Y(:,i) = Y(:,i) - grand_mean' - (XZ(i,ind_preserve)*beta_hat(ind_preserve,:))';
  end
  Y = Y./(std_pooled*ones(1,n_array));

  % Get regression batch effect parameters
  if verbose, fprintf('[ComCAT] Fitting L/S model\n'); end
  X_nuisance = XZ(:,[ind_batch ind_nuisance]);
  gamma_hat_masked = pinv(X_nuisance)*Y';

  % Remove additive nuisance effects before estimating site-specific scales.
  Y_for_delta = Y;
  if n_Z
    Y_for_delta = Y_for_delta - (XZ(:,ind_nuisance)*gamma_hat_masked(ind_nuisance,:))';
  end
  
  delta_hat = zeros(n_batch+n_Z,size(Y,1));
  for i=1:n_batch
    indices = batches{i};
    if mean_only
      delta_hat(i,:) = ones(1,size(Y,1));
    else
      delta_hat(i,:) = var(Y_for_delta(:,indices),[],2)';
    end
  end
  for i=n_batch+1:n_batch+n_Z
    varY = var(Y_for_delta,[],2)';
    if mean_only
      delta_hat(i,:) = ones(1,size(Y,1));
    else
      delta_hat(i,:) = varY;
    end
  end
  clear Y_for_delta
  
  if verbose, fprintf('[ComCAT] Adjusting the Data\n'); end
  for i=1:n_batch
    indices = batches{i};
    denom = sqrt(delta_hat(i,:))'*ones(1,n_batches(i));
    numer = (Y(:,indices)-(X_nuisance(indices,:)*gamma_hat_masked)');
    Y(:,indices) = numer./denom;
  end
  
  % only run this if no batch was defined
  if ~n_batch && n_Z
    denom = sqrt(delta_hat)'*ones(1,n_array);
    numer = (Y-(X_nuisance*gamma_hat_masked)');
    Y = numer./denom;
  end
  clear denom numer
  
  % ensure that no Inf values exist
  Y(~isfinite(Y)) = 0;

  % return indexed data and rescue NaNs for ComCAT
  Y_harmonized = zeros(dim,'single');
  for i=1:n_array
    Y_harmonized(ind_mask,i) = (Y(:,i).*std_pooled) + grand_mean' + (XZ(i,ind_preserve)*beta_hat(ind_preserve,:))';
  end  
  Y_harmonized(ind_NaN,:) = NaN;
  
  if nargout > 2
    ng = size(gamma_hat_masked,1);
    gamma_hat = zeros(ng,dim(1));
    for i=1:ng
      gamma_hat(i,ind_mask) = gamma_hat_masked(i,:); 
    end
  end
  
  if nargout > 3
      nd = size(delta_hat,1);
      delta_hat_masked = delta_hat;
      delta_hat = zeros(nd,dim(1));
      for i=1:nd
        delta_hat(i,ind_mask) = delta_hat_masked(i,:); 
      end
  end
  
  % transpose result if Y was transposed
  if transp, Y_harmonized = Y_harmonized'; end
end

%-------------------------------------------------------------------------------
function y = polynomial(x,p)
% Polynomial expansion and orthogonalization of function x
% FORMAT y = polynomial(x,p)
% x   - data matrix
% p   - order of polynomial [default: 1]
% 
% y   - orthogonalized data matrix
%__________________________________________________________________________
%
% polynomial orthogonalizes a polynomial function of order p
% ______________________________________________________________________
%
% Christian Gaser, Robert Dahnke
% Structural Brain Mapping Group (http://www.neuro.uni-jena.de)
% Departments of Neurology and Psychiatry
% Jena University Hospital
% ______________________________________________________________________

  if nargin < 2, p = 1; end
  
  if size(x,1) < size(x,2)
      x = x';
  end
  
  y = spm_detrend(x(:));
  v = zeros(size(y,1),p + 1);
  
  for j = 0:p
      v(:,(j + 1)) = (y.^j) - v*(pinv(v)*(y.^j));
  end
  
  for j = 2:p
      y = [y v(:,(j + 1))];
  end
end

