function test_comcat(test)

if ~nargin
  test = 1;
end

n = 400; V = 2000;
rng(1)

mean_only = 1;

switch test
case 1
% Additive linear nuisance, no batch. This checks the core continuous extension 
% in its strongest use case. After correction, association with Z should be removed 
% while association with X is preserved.

  X = randn(n,1);
  Z = randn(n,1);
  E = randn(V,n);
  Y = 2*X' + 1.5*Z' + E;     % V x n
  
  Yh = comcat(Y, [], Z, X, mean_only, 1, 0);   % mean_only=1
  
  % Diagnostics
  rZ_before = mean(abs(corr(Y', Z)));
  rZ_after  = mean(abs(corr(Yh', Z)));
  rX_before = mean(abs(corr(Y', X)));
  rX_after  = mean(abs(corr(Yh', X)));
  disp([rZ_before rZ_after rX_before rX_after])
case 2
% Additive nonlinear nuisance with polynomial basis. This checks the actual 
% polynomial extension used in comcat.m.

  X = randn(n,1);
  Z = linspace(-1,1,n)';
  E = randn(V,n);
  Y = 2*X' + 1.0*Z' + 2.0*(Z.^2)' + E;
  
  Yh1 = comcat(Y, [], Z, X, mean_only, 1, 0);   % linear nuisance only
  Yh2 = comcat(Y, [], Z, X, mean_only, 2, 0);   % quadratic nuisance basis
  
  % Compare residual association with Z and Z.^2
  rZ1  = mean(abs(corr(Yh1', Z)));
  rZ2  = mean(abs(corr(Yh2', Z)));
  rZ21 = mean(abs(corr(Yh1', Z.^2)));
  rZ22 = mean(abs(corr(Yh2', Z.^2)));

case 3
% Continuous multiplicative nuisance. This test should currently fail, and that 
% is precisely the point.

  X = randn(n,1);
  Z = randn(n,1);
  E = randn(V,n);
  sigma = exp(0.6*Z');             % nuisance-dependent scale
  Y = 2*X' + 1.0*Z' + sigma .* E;
  
  Yh = comcat(Y, [], Z, X, mean_only, 1, 0);   % scale enabled
  
  % Residuals after preserving X
  BetaX = (pinv([X ones(n,1)]) * Yh')';
  R = Yh - BetaX(:,1) * X';
  
  % Check whether variance still depends on Z
  lz = mean(log(var(R,[],1) + eps)); %#ok<NASGU>
  disp(lz)
  % Better: regress log feature-wise squared residuals on Z per feature subset

end