



% 
% Plot Figure 2c Distribution of Cells as a function of radius
% 
% 
% This code generates 240,000 neurons and 240,000 glial cells in a 739.81 radius network
% It is intended to plot the distribution of cells as a function of radius
% 
% 
% Input files
% 
% ./neuron_pos.csv      Neuron Positions
% ./glia_pos.csv        Glial Cell Positions





clear all;
close all;
fontsz=16;

glia_pos = readmatrix('glia_pos.csv');
neuron_pos = readmatrix('neuron_pos.csv');


glia_r=vecnorm(glia_pos',2);
neuron_r=vecnorm(neuron_pos',2);
rs=739.81;
edges_r=linspace(0,rs,21);
glia_num=histcounts(glia_r,edges_r);
neuron_num=histcounts(neuron_r,edges_r);
delta_l=diff(edges_r(1:2));
mid_r=(edges_r(1:end-1)+edges_r(2:end))/2;

cell_num=glia_num+glia_num;
NT=sum(cell_num);


colormap('jet');

% x_array=neuron_array(:,1);
neuron_fit=polyfit(mid_r,neuron_num,2);
glia_fit=polyfit(mid_r,glia_num,2);
cell_fit=polyfit(mid_r,cell_num,2);
cell_num_theory=(3*NT*delta_l/rs^3)*mid_r.^2;

neuron_pct=100*neuron_num./cell_num;

figure(1);clf;

yyaxis left
h=bar(mid_r,[neuron_num;glia_num]','stacked');
set(h(1),'FaceColor','g');
set(h(2),'FaceColor','b');

hold on;
plot(mid_r,cell_num_theory,'r-',...
    mid_r,polyval(cell_fit,mid_r),'y:',...
    'LineWidth',2);


set(gca,'FontSize',fontsz,'YColor','k');
xlabel('M/r_n','FontSize',fontsz);
ylabel('Num. Cells','FontSize',fontsz);
xlim(minmax(edges_r));
ylim([0, max(cell_num)*1.1]);

yyaxis right
plot(mid_r,neuron_pct,'k-o','LineWidth',2,'MarkerSize',8);
legend('Neuron','Glial','Theory','Fit', 'Neuron/Cell','FontSize',fontsz-2,'Location','NorthWest','Box','off','NumColumns',2);
set(gca,'FontSize',fontsz,'YColor','k');
ylabel('Neuron Percentage (%)','FontSize',fontsz);
ylim([0,100]);
saveas(gcf,'density_init','fig');
saveas(gcf,'density_init','png');

