
% 
% Plot Figure 2a Measuring the Cell Density and Probability of Collisions
% The neural network sphere radius is constant, while the number of cells changes.
% It allows us to plot cell density vs the probability of cell collisions
% 
% Input files
% ./initial_cell_num.csv       Contains the initial number of cells
% ./final_neuron_num.csv       Contains the final number of neurons after deleting collided cells
% ./final_glia_num.csv         Contains the final number of glial cells after deleting collided cells
% ./collision_run_time.csv     Contains the time it takes to run collision detection




clear all;
close all;

fontsz=16;




% Load Data
initial_size = readmatrix('initial_cell_num.csv');
final_neuron_num = readmatrix('final_neuron_num.csv');
final_glia_num = readmatrix('final_glia_num.csv');
computation_time = readmatrix('collision_run_time.csv');




% Format Data
initial_size = repmat(initial_size,1,10);
initial_size = reshape(initial_size,23,10);
final_neuron_num = reshape(final_neuron_num,23,10);
final_glia_num = reshape(final_glia_num,23,10);
computation_time = reshape(computation_time,23,10);

initial_size = initial_size(2:end,:);
final_neuron_num = final_neuron_num(2:end,:);
final_glia_num = final_glia_num(2:end,:);
computation_time = computation_time(2:end,:);



computation_time_std = std(computation_time,0,2);
computation_time = mean(computation_time,2);

% Compute Density

total_num = final_glia_num + final_neuron_num;
density = initial_size./9.047E8;
probofcollisions = (initial_size - total_num) ./ initial_size;



initial_size = mean(initial_size,2);
probofcollisions_std = std(probofcollisions,0,2);
probofcollisions = mean(probofcollisions,2);


density=density(probofcollisions>0);
probofcollisions_std=probofcollisions_std(probofcollisions>0);
probofcollisions=probofcollisions(probofcollisions>0);

collision_fit=polyfit(log(density),log(probofcollisions),1);
probofcollisions_fit=exp(polyval(collision_fit,log(density)));
initial_size_fit=polyfit(log(initial_size),log(computation_time),1);
computation_time_fit=exp(polyval(initial_size_fit,log(initial_size)));
probofcollisions_theory=(32*pi/3)*density;
figure(1);clf;
hold on
errorbar(density,probofcollisions,probofcollisions_std,'.','LineWidth',2,'MarkerSize',10,'Color','b');
plot(density,probofcollisions_fit,'b--',density,probofcollisions_theory,'r-','LineWidth',2,'MarkerSize',10);
set(gca,'XScale','log','YScale','log','FontSize',fontsz);
xlabel("Density (r_n^{-3})",'FontSize',fontsz);
ylabel("Probability of Collision",'FontSize',fontsz);
legend('Data',['Fit (Slope: ', num2str(collision_fit(1),'%3.2f'), ')'],'Theory','FontSize',fontsz-2,'Location','NorthWest','box','off');
hold off
drawnow;
saveas(gcf,'collisions','fig');
saveas(gcf,'collisions','png');




