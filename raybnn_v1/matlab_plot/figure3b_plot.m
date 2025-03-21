clear all;
close all;
fontsz=16;

WValues = readmatrix("before_delete_WValues.csv");

w_minmax=minmax(WValues');
w_max=max(abs(w_minmax));
num_bins=200;
[w_counts,w_edges]=histcounts(WValues,num_bins);
w_centers=(w_edges(1:end-1)+w_edges(2:end))/2;
w_centers=w_centers/w_max;
% idx=(abs(w_centers)<0.2);
idx=1:length(w_centers);
fit_model=fit(w_centers(idx)',w_counts(idx)','gauss1');

figure(1);clf;
h=bar(w_centers,w_counts);
hold on;
plot(w_centers,fit_model(w_centers),'r-','LineWidth',2);
legend("Counts","Fit (\sigma="+num2str(fit_model.c1/sqrt(2),'%.3f')+")",'FontSize',fontsz);
set(gca,'FontSize',fontsz);
% xlim([-4.0*fit_model.c1, 4*fit_model.c1])
xlim([-1,1]);
ylabel('Counts','FontSize',fontsz);
xlabel('W/W_{max}','FontSize',fontsz);
saveas(gcf,"WeightCounts",'fig');
saveas(gcf,"WeightCounts",'png');


