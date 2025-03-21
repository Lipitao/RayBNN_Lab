



% 
% Plot  2f Probability Density Function of the Number of Connections per Neuron
% 
% This code uses RT-3 to plot the probability density function of the number of neural connections per neuron compared to the density of the neural network sphere
% 
% Input files
% ./WRowIdxCOO_*.csv     Row index of the COO Sparse Matrix
% ./WColIdx_*.csv        Column index of the COO Sparse Matrix
% ./neuron_pos_*.csv     Neuron positions
% ./neuron_idx_*.csv     Indexes of the neurons
% 






clear all;
close all;
fontsz=16;



% Load Density file
density_list = readmatrix("fig2f_density_list.csv");

density_num = length(density_list);
bucket_size = 20;
connumarr = zeros(density_num,bucket_size);
probarr = zeros(density_num,bucket_size);




for ii=1:length(density_list)
density_num = density_list(ii);
if density_num >= 1.0
    density_num = uint32(density_num);
end

% Load More files
filename = strcat('fig2f_WColIdx_',sprintf('%.5f',density_num),'.csv');
WColIdx = readmatrix(filename,'OutputType','uint32');

filename = strcat('fig2f_WRowIdxCOO_',sprintf('%.5f',density_num),'.csv');
WRowIdxCOO = readmatrix(filename,'OutputType','uint32');

filename = strcat('fig2f_neuron_idx_',sprintf('%.5f',density_num),'.csv');
neuron_idx = readmatrix(filename,'OutputType','uint32');


filename = strcat('fig2f_neuron_pos_',sprintf('%.5f',density_num),'.csv');
neuron_pos = readmatrix(filename);



WRowIdx = uint32(WRowIdxCOO)+1;
WColIdx = uint32(WColIdx)+1;
neuron_idx = uint32(neuron_idx)+1;




WRowIdx = uint32(WRowIdxCOO)+1;
WColIdx = uint32(WColIdx)+1;
neuron_idx = uint32(neuron_idx)+1;


% Compute the number of connections per neuron

[~,~,ix] = unique(WRowIdx);
C = accumarray(ix,1).';

[prob,connum] = hist(C,20);

connumarr(ii,:) = connum;
probarr(ii,:) = prob/trapz(prob);

end


% Plot Probability Density Function of Number of Connections per Neuron


figure(1);clf;

for count=1:length(density_list)
    plot3(density_list(count)*ones(1,size(connumarr,2)),connumarr(count,:),probarr(count,:),'-','LineWidth',2);
    hold on;
    grid on;
end
xlabel('Density (r_n^{-3})','FontSize',fontsz,'rotation',19,'VerticalAlignment','top','HorizontalAlignment','left');
ylabel('Connections','FontSize',fontsz,'rotation',-25,'VerticalAlignment','top','HorizontalAlignment','right');
zlabel('Probability','FontSize',fontsz);
set(gca,'FontSize',fontsz,'XScale','log');


drawnow;
%xlim(minmax(x_array));
%ylim(minmax(y_array));
saveas(gcf,'NeuralConNum','fig');
saveas(gcf,'NeuralConNum','png');


