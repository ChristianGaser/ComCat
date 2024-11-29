function simulate_comcat_plot

n = 1000;
mean_only = 1;
apply_2step_correction = 1; % Correction by Zhao

if mean_only, str = '_mean_only'; else str = ''; end
if ~apply_2step_correction, str = [str '_nocorr']; end
name = sprintf('D_FPR_comcat_n%d%s.mat',n, str);
load(name)

n1 = numel(a2);
n2 = numel(a4);

xl = cell(n1,1);
yl = cell(n2,1);
for i=1:n1
  xl{i} = num2str(a2(i));
end
for i=1:n2
  yl{i} = num2str(a4(i));
end

D1=squeeze(D(:,:,:,1)); % Ancova
D2=squeeze(D(:,:,:,2)); % ComCat
FPR1=squeeze(FPR(:,:,:,1)); % Ancova
FPR2=squeeze(FPR(:,:,:,2)); % ComCat

mn_FPR = 0.04;
mx_FPR = 0.1;

for j=1:numel(n_nuisance)
  f = figure(11+j);
  imagesc(squeeze(FPR2(:,:,j)))
  title(['FPR ComCat #nuisance=' num2str(n_nuisance(j))])
  axis image
  set(gca,'XTick',1:n2,'XTicklabel',yl,'YTick',1:n1,'YTickLabel',xl)
  xlabel('Correlation nuisance-X')
  ylabel('Correlation nuisance-Y')
  colorbar
  clim([mn_FPR mx_FPR])
  colormap(jet)
  if mean_only
    saveas(gcf,['FPR_Comcat_' num2str(n_nuisance(j)) '_meanonly.png']);
  else
    saveas(gcf,['FPR_Comcat_' num2str(n_nuisance(j)) '.png']);
  end
end
