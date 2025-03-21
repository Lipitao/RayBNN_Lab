

% 
% Plot Figure 2d Runtimes of Various Raytracing Algorithms
% 
% 
% This code benchmarks the runtimes of RT-1,RT-2, and RT-3.
% RT-3 has variable 20, 40, and 60 neuron radii
% 
% 
% Input files
% 
% ./RT1_run_time.csv       List of time benchmarks for RT1 algorithm
% ./RT2_run_time.csv       List of time benchmarks for RT2 algorithm 
% ./RT3_20_run_time.csv    List of time benchmarks for RT3 algorithm with neuron radius 20
% ./RT3_40_run_time.csv    List of time benchmarks for RT3 algorithm with neuron radius 40
% ./RT3_60_run_time.csv    List of time benchmarks for RT3 algorithm with neuron radius 60
% ./neuron_num_list.csv    List of neuron sizes




clear all
close all
fontsz=16;



%Load files

neuron_num = readmatrix('neuron_num_list.csv');
RT1_array = readmatrix('RT1_run_time.csv');
RT2_array = readmatrix('RT2_run_time.csv');
RT3_20_array = readmatrix('RT3_20_run_time.csv');
RT3_40_array = readmatrix('RT3_40_run_time.csv');
RT3_60_array = readmatrix('RT3_60_run_time.csv');

RT1_len = size(RT1_array,2);
RT2_len = size(RT2_array,2);

% Linear fit to plot

RT1_fit=polyfit(log(neuron_num(1:RT1_len)),log(RT1_array),1);
RT2_fit=polyfit(log(neuron_num(1:RT2_len)),log(RT2_array),1);
RT3_20_fit=polyfit(log(neuron_num),log(RT3_20_array),1);
RT3_40_fit=polyfit(log(neuron_num),log(RT3_40_array),1);
RT3_60_fit=polyfit(log(neuron_num),log(RT3_60_array),1);

% Plot the Raytracing algorithm vs computation time

figure(1);clf;
h=plot(neuron_num(1:RT1_len),RT1_array,'r+',...
    neuron_num(1:RT1_len),exp(polyval(RT1_fit,log(neuron_num(1:RT1_len)))),'r-',...
    neuron_num(1:RT2_len),RT2_array,'bo',...
        neuron_num(1:RT2_len),exp(polyval(RT2_fit,log(neuron_num(1:RT2_len)))),'b-',...
    neuron_num,RT3_20_array,'ks',...
            neuron_num,exp(polyval(RT3_20_fit,log(neuron_num))),'k-', ...
    neuron_num,RT3_40_array,'gs',...
            neuron_num,exp(polyval(RT3_40_fit,log(neuron_num))),'g-', ...
    neuron_num,RT3_60_array,'ms',...
            neuron_num,exp(polyval(RT3_60_fit,log(neuron_num))),'m-', ...
            'linewidth',2,'markersize',8);
set(gca,'xscale','log','yscale','log','FontSize',fontsz);
legendstr={'RT-1','', 'RT-2','', 'RT-3 20r_n','', 'RT-3 40r_n','', 'RT-3 60r_n',''};
legend(legendstr,'Location','NorthWest','FontSize',fontsz-2,'Box','off','NumColumns',2);
xlabel('Number of Neurons','FontSize',fontsz);
ylabel('Raytracing time (s)','FontSize',fontsz);
saveas(gcf,'raytracing','fig');
saveas(gcf,'raytracing','png');