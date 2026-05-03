% gen_test_data.m
% Generate test cases, run comcat.m, and save inputs + outputs as .mat files
% so that comcat.py can be validated against the same data.
%
% Run this once from MATLAB before running test_comcat_py.py.

addpath('/Users/gaser/spm/spm12');

rng(42);

n = 120;   % subjects
V = 500;   % features / voxels

% -------------------------------------------------------------------
% Case 1: multi-site, additive nuisance, preserve covariate
% -------------------------------------------------------------------
batch1  = [ones(1,40) 2*ones(1,40) 3*ones(1,40)]';
Z1      = randn(n,1);                           % nuisance (age-like)
X1      = randn(n,1);                           % preserve (group-like)
E1      = randn(V,n);
% site offsets
site_off = [0; 2; -2];
Y1 = 3*X1' + 1.5*Z1' + site_off(batch1)' + E1;

[Yh1, bh1, gh1, dh1] = comcat(Y1, batch1, Z1, X1, 0, 2, 0);

save('test_case1.mat', 'Y1','batch1','Z1','X1','Yh1','bh1','gh1','dh1','-v6');
fprintf('Saved test_case1.mat\n');

% -------------------------------------------------------------------
% Case 2: no batch (single site), polynomial nuisance removal
% -------------------------------------------------------------------
Z2 = linspace(-1,1,n)';
X2 = randn(n,1);
E2 = randn(V,n);
Y2 = 2*X2' + 1.0*Z2' + 2.0*(Z2.^2)' + E2;

[Yh2, bh2, gh2, dh2] = comcat(Y2, [], Z2, X2, 1, 2, 0);

save('test_case2.mat', 'Y2','Z2','X2','Yh2','bh2','gh2','dh2','-v6');
fprintf('Saved test_case2.mat\n');

% -------------------------------------------------------------------
% Case 3: mean_only mode, two sites, no nuisance
% -------------------------------------------------------------------
batch3 = [ones(1,60) 2*ones(1,60)]';
E3     = randn(V,n);
X3     = randn(n,1);
Y3     = 2*X3' + repmat([zeros(1,60) 5*ones(1,60)], V, 1) + E3;

[Yh3, bh3, gh3, dh3] = comcat(Y3, batch3, [], X3, 1, 1, 0);

save('test_case3.mat', 'Y3','batch3','X3','Yh3','bh3','gh3','dh3','-v6');
fprintf('Saved test_case3.mat\n');

% -------------------------------------------------------------------
% Case 4: multi-site, polynomial degree=3, nuisance + preserve
% -------------------------------------------------------------------
batch4 = [ones(1,40) 2*ones(1,40) 3*ones(1,40)]';
Z4     = linspace(-1,1,n)';            % structured nuisance
X4     = randn(n,1);                   % preserve
E4     = randn(V,n);
site_off4 = [0; 3; -3];
Y4 = 2*X4' + Z4' + 1.5*(Z4.^2)' + 0.8*(Z4.^3)' + site_off4(batch4)' + E4;

[Yh4, bh4, gh4, dh4] = comcat(Y4, batch4, Z4, X4, 0, 3, 0);

save('test_case4.mat', 'Y4','batch4','Z4','X4','Yh4','bh4','gh4','dh4','-v6');
fprintf('Saved test_case4.mat\n');

% -------------------------------------------------------------------
% Case 5: four sites, two nuisance columns, preserve, poly_degree=3
% -------------------------------------------------------------------
batch5 = [ones(1,30) 2*ones(1,30) 3*ones(1,30) 4*ones(1,30)]';
Z5a    = randn(n,1);                   % nuisance col 1
Z5b    = linspace(-1,1,n)';            % nuisance col 2
Z5     = [Z5a Z5b];
X5     = randn(n,1);                   % preserve
E5     = randn(V,n);
site_off5 = [0; 2; -2; 4];
Y5 = 3*X5' + Z5a' + 0.5*(Z5b.^2)' + site_off5(batch5)' + E5;

[Yh5, bh5, gh5, dh5] = comcat(Y5, batch5, Z5, X5, 0, 3, 0);

save('test_case5.mat', 'Y5','batch5','Z5','X5','Yh5','bh5','gh5','dh5','-v6');
fprintf('Saved test_case5.mat\n');

fprintf('Done. Now run:  python test_comcat_py.py\n');
