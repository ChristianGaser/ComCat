function simulate_comcat_ui

a1=0;
a2=0:0.05:0.3;
a4=0:0.05:0.5;
no_preserving = 0;
n = 1000;
n_sim = 20000;
n_nuisance = [1 2 5 10];
mean_only = 1;

D   = zeros(numel(a2),numel(a4),numel(n_nuisance),2);
FPR = zeros(numel(a2),numel(a4),numel(n_nuisance),2);
for j=1:numel(a2)
  for k=1:numel(a4)
    for m=1:numel(n_nuisance)
      fprintf('\n------------------------------------------------------------------\n');
      fprintf('%g\t%g\t%d\t%d',a2(j),a4(k),n_nuisance(m));
      a = [a1 a2(j) 0 a4(k)];
      [D(j,k,m,:), FPR(j,k,m,:)] = simulate_comcat(a, no_preserving, ...
          n, n_sim, n_nuisance(m),0,mean_only,1);
    end
  end
end

if mean_only, str = '_mean_only'; else str = ''; end
name = sprintf('D_FPR_comcat_n%d%s_nocorr.mat',n, str);
save(name)
