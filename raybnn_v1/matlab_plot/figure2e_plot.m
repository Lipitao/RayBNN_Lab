


% 
% Plot Figure 2e Probability Density Function of the Ray Lengths
% 
% 
% This code uses RT-3 to plot the probability density function of the raylengths
% compared to the density of the neural network sphere
% 
% 
% Input files
% 
% ./WRowIdxCOO_*.csv     Row index of the COO Sparse Matrix
% ./WColIdx_*.csv        Column index of the COO Sparse Matrix
% ./neuron_pos_*.csv     Neuron positions
% ./neuron_idx_*.csv     Indexes of the neurons









clear all;
close all;
fontsz=16;

% Load Density file
density_list = readmatrix("density_list.csv");

density_num = length(density_list);
bucket_size = 20;
conlengtharr = zeros(density_num,bucket_size);
probarr = zeros(density_num,bucket_size);




for ii=1:length(density_list)
density_num = density_list(ii);
if density_num >= 1.0
    density_num = uint32(density_num);
end

% Load More files
filename = strcat('WColIdx_',sprintf('%.5f',density_num),'.csv');
WColIdx = readmatrix(filename,'OutputType','uint32');

filename = strcat('WRowIdxCOO_',sprintf('%.5f',density_num),'.csv');
WRowIdxCOO = readmatrix(filename,'OutputType','uint32');

filename = strcat('neuron_idx_',sprintf('%.5f',density_num),'.csv');
neuron_idx = readmatrix(filename,'OutputType','uint32');


filename = strcat('neuron_pos_',sprintf('%.5f',density_num),'.csv');
neuron_pos = readmatrix(filename);



WRowIdx = uint32(WRowIdxCOO)+1;
WColIdx = uint32(WColIdx)+1;
neuron_idx = uint32(neuron_idx)+1;



neuron_size = max(neuron_idx);

total_neuron_pos = zeros(neuron_size,3);

total_neuron_pos(neuron_idx,:) =  neuron_pos;

%Compute Ray Length

connections = size(WRowIdx,1);
d = zeros(connections,1);

for i = 1:connections
    t = WRowIdx(i);
    s = WColIdx(i);
    
    pt = total_neuron_pos(t,:);
    ps = total_neuron_pos(s,:);
    
    d(i) = norm(pt - ps);
    
end


g = d;
g = g(g > 0.00001); 

%Compute Histogram

conlength = (0:4:76)  + 2;

[prob,conlength] = hist(g,conlength);

conlengtharr(ii,:) = conlength;
probarr(ii,:) = prob/trapz(conlength,prob);
end




% Plot Probability Density Function of Ray Lengths


[r_array,p_array]=Prob_length(density_list);
color_mat=colororder;
figure(1);clf;

for count=1:size(probarr,1)
    coloridx=mod(count,size(color_mat,1))+1;
    plot3(density_list(count)*ones(1,size(probarr,2)),conlengtharr(count,:),probarr(count,:),...
        'Color',color_mat(coloridx,:),'Marker','+','LineStyle','none','MarkerSize',10);
    hold on;
    grid on;
    plot3(density_list(count)*ones(size(r_array)),r_array,p_array{count},'Color',color_mat(coloridx,:),'LineStyle','-','LineWidth',2);
    disp("count="+count+"total: "+sum(probarr(count,:))*diff(conlengtharr(count,1:2)));
end

xlabel('Density (r_n^{-3})','FontSize',fontsz,'rotation',19,'VerticalAlignment','top','HorizontalAlignment','left');
ylabel('r/r_n','FontSize',fontsz,'rotation',-25,'VerticalAlignment','top','HorizontalAlignment','right');
zlabel('Probability','FontSize',fontsz);
xlim([min(density_list),0.01])
set(gca,'FontSize',fontsz,'XScale','log');
%ylim(minmax(y_array));
saveas(gcf,'NeuralConDist','fig');
saveas(gcf,'NeuralConDist','png');
