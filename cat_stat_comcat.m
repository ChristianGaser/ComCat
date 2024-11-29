function [Y_adjusted, gamma_hat, delta_hat] = cat_stat_comcat(Y, batch, nuisance, preserve, mean_only, poly_degree, verbose)
% ComCAT harmonization for sites and nuisance parameters
% ComCAT: Combating CovariATe effects
% Format: Y_adjusted = cat_stat_comcat(Y, batch, nuisance, preserve, mean_only, poly_degree, verbose)
% ______________________________________________________________________
%
% Input:
% Y           - data matrix with size n_data x n_subjects
%               or char array of filenames
% batch       - vector of site coding with size n_data x 1
% nuisance    - matrix or vector of parameter(s) that should be removed (size n_data x n_parameters)
% preserve    - matrix or vector of parameter(s) that should be preserved (size n_data x n_parameters)
% mean_only   - only mean adjusted (no scaling) (default no)
% poly_degree - degree of polynomial extension of nuisance parameter(s) (default 2)
% verbose     - be verbose (default no)
%
% Output:
% Y_adjusted  - adjusted data matrix with size n_data x n_subjects
%
% Harmonized data is stored in a separate subfolder whose name can be specified.
% Optionally the LS estimates can be saved for gamma (estimates for additive effects) and 
% delta (estimates for multiplicative effects).
%
% Heavily modified and extended version of ComBat:
% https://github.com/Jfortin1/ComBatHarmonization/tree/master/Matlab
% ______________________________________________________________________
%
% Christian Gaser, Robert Dahnke
% Structural Brain Mapping Group (http://www.neuro.uni-jena.de)
% Departments of Neurology and Psychiatry
% Jena University Hospital
% ______________________________________________________________________
% $Id: cat_stat_comcat.m 1868 2021-07-29 15:21:21Z gaser $

% call interactive mode
if nargin == 0 || ischar(Y)
  if nargin && ischar(Y)
    P = Y;
  else
    P = spm_select(Inf,{'mesh','image','mat','txt'},'Select spatially registered data to harmonize');
  end
  name1 = deblank(P(1,:));
  mesh_detected = spm_mesh_detect(name1);
  [pth,nam,ext] = spm_fileparts(name1);
  
  % use faster nifti function for reading data
  if mesh_detected
    filetype = 2; % gifti
    V = spm_data_hdr_read(P);
  else
    if strcmp(ext,'.mat')
      filetype = 3; % mat-file
    elseif strcmp(ext,'.txt')
      filetype = 4; % txt-file
    else 
      filetype = 1; % nifti
      V = nifti(P);
    end
  end

  switch filetype
  % nifti/gifti
  case {1,2}
    n = numel(V);
  case 3 % mat-file
    % only one mat-file allowed
    if size(P,1) > 1
      fprintf('Only one file allowed. Ignore other files!\n');
    end
    
    % load first file and check for Y field
    V = load(name1);
    if ~isfield(V,'Y')
      error('Mat-file does not contain Y data field.')
    end
    n = size(V.Y,1);
  case 4 % txt-file
    % only one txt-file allowed
    if size(P,1) > 1
      fprintf('Only one file allowed. Ignore other files!\n');
    end
    
    % load first file and check for Y field
    V0 = spm_load(name1);
    n = size(V0,1);
    V.Y = V0;
  end
  
  if nargin < 2
    do_batch = spm_input('Multi-site data?',1,'yes|no',[1 0],1);
    if do_batch
      batch = spm_input('Site coding',1,'r',[],[1,n]);
    else
      batch = ones(n,1);
    end
  end
  n_sites = max(batch);
  
  if n_sites > 1
    do_batch = 1;
  else
    do_batch = 0;
  end

  if nargin < 3
    done = 0;
    while ~done
      nuisance = spm_input('Nuisance parameter (0 to skip)','+1','r');
      if numel(nuisance) == 1 && nuisance == 0
        nuisance = [];
        done = 1;
      else
        if size(nuisance,1) == n
          done = 1;
        elseif size(nuisance,2) == n
          nuisance = nuisance';
          done = 1;
        end
      end
    end
  end
  n_nuisance = size(nuisance,2);
  
  if nargin < 4
    do_preserve = spm_input('Preserve a parameter?','+1','yes|no',[1 0],1);
    if do_preserve
      preserve = spm_input('Parameters to preserve','+1','r',[],[Inf,n]);
    else    
      preserve = [];
    end
  else
    do_preserve = ~isempty(preserve);
  end
  
  if nargin < 5
    mean_only = spm_input('Correct for mean (location) only?',1,'yes|no',[1 0],0);
  end
  
  % offer polynomial extension with degree 1, 2, or 3
  if nargin < 6
    if n_nuisance
      poly_degree = spm_input('Polynomial extension for nuisance?','+1','linear|quadratic|cubic',[1 2 3],1);
    else
      poly_degree = 1;
    end
  end
      
  % estimates will be not saved for mat-files
  if filetype ~= 3 && nargin == 0
    save_estimates = spm_input('Save LS estimates (gamma/delta)?','+1','yes|no',[1 0],2);
  else
    save_estimates = 0;
  end
  
  % use subfolders for nifti/gifti-files
  if do_batch && ~n_nuisance
    subfolder = 'combat';
  else
    subfolder = 'comcat';
  end
  
  if do_batch,    subfolder = [subfolder sprintf('_sites%d',n_sites)]; end
  if do_preserve, subfolder = [subfolder sprintf('_preserve%d',size(preserve,2))]; end
  if mean_only,   subfolder = [subfolder '_meanonly']; end
  if n_nuisance
    if poly_degree > 1
      subfolder = [subfolder sprintf('_nuisance%d_poly%d',n_nuisance,poly_degree)];
    else
      subfolder = [subfolder sprintf('_nuisance%d',n_nuisance)];
    end
  end
  
  if filetype < 3
    if nargin == 0
      subfolder = spm_input('Subfolder for harmonized data?',1,'s',subfolder);
    end
  else
    % use 'h' to indicate harmonization for mat/txt-files
    prep_str = 'h';
    % use subfolder as log-filename
    subfolder = spm_input('Name for Comcat log-file?',1,'s',subfolder);
  end
  
  % check whether subfolder already exists
  if filetype < 3 && exist(fullfile(pth,subfolder),'dir')
    overwrite = spm_input('Overwrite existing subfolder?','+1','yes|no',[1 0],1);
    if ~overwrite
      subfolder = spm_input('Subfolder for harmonized data?',1,'s',subfolder);
    end
  end

  % nifti/gifti
  if filetype < 3
    Y = zeros(prod(V(1).dat.dim),n,'single');
    fprintf('Read data ');
    for i=1:n
      fprintf('.')
      if mesh_detected
        tmp = spm_data_read(V(i));
      else
        tmp(:,:,:) = single(V(i).dat(:,:,:));
      end
      Y(:,i) = tmp(:);
    end
    fprintf('\n');
  else % mat/txt-file
    Y = V.Y;
  end
  
  verbose = 1;
  
end

% default values
if ~exist('verbose','var'),     verbose = 1; end
if ~exist('poly_degree','var'), poly_degree = 3; end
if ~exist('mean_only','var'),   mean_only = 0; end
if ~exist('preserve','var'),    preserve = []; end

% define batch as intercept if empty
if isempty(batch)
  batch = ones(size(nuisance,1),1);
end

% ensure that site coding begins with 1 and is increasing
[tmp, tmp2, batch] = unique(batch);

n = numel(batch);

% transpose preserve if needed
if size(preserve,1) ~= n
  preserve = preserve';
end

% check whether we have to transpose Y
[x,y] = size(Y);
transp = 0;
if y ~= n
  if x == n
    transp = 1;
    Y = Y';
  else
    error('Size mismatch between data (%dx%d) and batch (%d)',x,y,n);
  end
end

% print out some correlations between nuisance and X and Y
if verbose
  cc = corrcoef([preserve nuisance]);
  fprintf('Correlation of nuisance variable with preserving variable (EoI): %5.3f\n',mean(abs(cc(1,2:end))));

end

[Y_adjusted, beta_hat, gamma_hat, delta_hat] = comcat(Y, batch, nuisance, preserve, mean_only, poly_degree, verbose);

% find and replace values that changed probably too much (SD changed by factor of 10)
sd0 = std(Y,[],2);
sd1 = std(Y_adjusted,[],2);
ind_extreme = (sd1./(sd0+eps)) > 10;
if sum(ind_extreme)
  fprintf('Changed %d voxels since values changed too much',sum(ind_extreme));
  Y_adjusted(ind_extreme,:) = Y(ind_extreme,:);
end

if transp
  Y_adjusted = Y_adjusted';
end
    
% write data if interactively defined 
if nargin == 0 || exist('P')

  % save some information about used parameters in mat-file
  if do_batch, batch0 = batch; else batch0 = []; end
  Comcat = struct('batch',batch0,'nuisance',nuisance','preserve',preserve,'poly_degree',poly_degree);
  
  matname = fullfile(pth,subfolder);
  save(matname,'Comcat');
  fprintf('Save ComCat log-information in %s\n',matname);

  % nifti/gifti
  if filetype < 3
  
    fprintf('Save harmonized data in subfolder "%s" ',subfolder);
    
    % save delta/gamma LS estimates
    if save_estimates
    
      for i=1:size(gamma_hat,1)
        gname = fullfile(pth,sprintf('gamma%02d%s',i,ext));
        Vg = V(1);
        
        if mesh_detected
          Vg.fname = gname;
          tmp = reshape(gamma_hat(i,:),Vg.dim);
          Vg = spm_data_hdr_write(Vg);
          spm_data_write(Vg,tmp);
        else
          Vg.dat.fname = gname;
          tmp = reshape(gamma_hat(i,:),Vg.dat.dim);
          create(Vg);
          Vg.dat(:,:,:) = tmp(:,:,:);
        end
        
      end
      
      for i=1:size(delta_hat,1)
        dname = fullfile(pth,sprintf('delta%02d%s',i,ext));
        Vd = V(1);
        
        if mesh_detected
          Vd.fname = dname;
          tmp = reshape(delta_hat(i,:),Vd.dim);
          Vd = spm_data_hdr_write(Vd);
          spm_data_write(Vd,tmp);
        else
          Vd.dat.fname = dname;
          tmp = reshape(delta_hat(i,:),Vd.dat.dim);
          create(Vd);
          Vd.dat(:,:,:) = tmp(:,:,:);
        end
        
      end
    end
          
    for i=1:n
    
      if mesh_detected
        fname = V(i).fname;
      else
        fname = V(i).dat.fname;
      end
      
      [pth1,nam1,ext1] = spm_fileparts(fname);
      fname = fullfile(pth1,subfolder,[nam1 ext1]);
      
      if ~exist(fullfile(pth1,subfolder),'dir')
        mkdir(fullfile(pth1,subfolder))
      end
      
      fprintf('.');
      
      if mesh_detected
        V(i).fname = fname;
        tmp = reshape(Y_adjusted(:,i),V(i).dim);
        V(i) = spm_data_hdr_write(V(i));
        spm_data_write(V(i),tmp);
      else
        tmp = reshape(Y_adjusted(:,i),V(i).dat.dim);
        V(i).dat.fname = fname;
        create(V(i));
        V(i).dat(:,:,:) = tmp(:,:,:);
      end
      
    end
  elseif filetype == 3 % mat-file
  
    fprintf('Save harmonized data with leading "%s" ',prep_str)
    V.Y = Y_adjusted;
    fname = fullfile(pth,[nam prep_str ext]);
    save(fname,'-struct','V');
    fprintf('Save %s\n',fname);
    
  else % txt-file
  
    fprintf('Save harmonized data with leading "%s" ',prep_str)
  
    % save delta/gamma LS estimates
    if save_estimates
    
      for i=1:size(gamma_hat,1)
        gname = fullfile(pth,sprintf('gamma%01d%s',i,ext));
        fid = fopen(gname,'w');
        fprintf(fid,'%g ',gamma_hat(i,:));
        fprintf(fid,'\n');
        fclose(fid);
        fprintf('Save %s (estimates for additive effects)\n',gname);
      end
      
      for i=1:size(delta_hat,1)
        dname = fullfile(pth,sprintf('delta%01d%s',i,ext));
        fid = fopen(dname,'w');
        fprintf(fid,'%g ',delta_hat(i,:));
        fprintf(fid,'\n');
        fclose(fid);
        fprintf('Save %s (estimates for multiplicative effects)\n',dname);
      end
    end
    
    fname = fullfile(pth,[prep_str nam ext]);
    fid = fopen(fname,'w');
    
    for i=1:n
      fprintf(fid,'%g ',Y_adjusted(i,:));
      fprintf(fid,'\n');
    end
    
    fclose(fid);
    fprintf('Save %s\n',fname);
  end
end

fprintf('\n')

% prevent that huge matrix is returned even if no output argument is defined
if ~nargout
 clear Y_adjusted gamma_hat delta_hat;
end

