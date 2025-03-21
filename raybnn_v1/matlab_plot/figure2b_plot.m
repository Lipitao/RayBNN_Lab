

% 
% Plot Figure 2b Measuring runtime of various collision detection algorithms
% 
% The code runs serial, mini-batch, and batch versions of cell collision detection.
% It compares the runtimes of those algorithms
% 
% 
% Input files
% 
% ./collision_run_time.csv           Contains the time it takes to run collision detection by looking at minibatches of cells
% ./collision_run_time_serial.csv    Contains the time it takes to run collision detection by looking at cells  one by one
% ./collision_run_time_batch.csv     Contains the time it takes to run collision detection by looking at all cells at once






clear all;
close all;

fontsz=16;




% Load Data
initial_size = readmatrix('initial_cell_num.csv');
computation_time = readmatrix('collision_run_time.csv');
computation_time_batch = readmatrix('collision_run_time_batch.csv');
computation_time_serial = readmatrix('collision_run_time_serial.csv');


initial_size = repmat(initial_size,1,10);
initial_size = reshape(initial_size,23,10);
computation_time = reshape(computation_time,23,10);
computation_time_batch  = reshape(computation_time_batch ,12,10);
computation_time_serial  = reshape(computation_time_serial ,18,10);


initial_size = initial_size(2:end,:);
computation_time = computation_time(2:end,:);
computation_time_batch = computation_time_batch(2:end,:);
computation_time_serial = computation_time_serial(2:end,:);

computation_time_std = std(computation_time,0,2);
computation_time = mean(computation_time,2);


computation_time_batch_std = std(computation_time_batch ,0,2);
computation_time_batch  = mean(computation_time_batch ,2);


computation_time_serial_std = std(computation_time_serial ,0,2);
computation_time_serial  = mean(computation_time_serial ,2);



density = initial_size./9.047E8;




initial_size = mean(initial_size,2);

initial_size_fit=polyfit(log(initial_size),log(computation_time),1);
computation_time_fit=exp(polyval(initial_size_fit,log(initial_size)));


initial_size_batch_fit=polyfit(log(initial_size(1:11)),log(computation_time_batch),1);
computation_time_batch_fit=exp(polyval(initial_size_batch_fit,log(initial_size(1:11))));


initial_size_serial_fit=polyfit(log(initial_size(1:17)),log(computation_time_serial),1);
computation_time_serial_fit=exp(polyval(initial_size_serial_fit,log(initial_size(1:17))));


figure(2);clf;
hold on

h=errorbar(initial_size(1:17),computation_time_serial,computation_time_serial_std,'.','LineWidth',2,'MarkerSize',10,'Color','r');
plot(initial_size(1:17),computation_time_serial_fit,'r','LineWidth',2,'MarkerSize',10);
fit0 = initial_size_serial_fit(1);


errorbar(initial_size(1:11),computation_time_batch,computation_time_batch_std,'.','LineWidth',2,'MarkerSize',10,'Color','g');
plot(initial_size(1:11),computation_time_batch_fit,'g-','LineWidth',2,'MarkerSize',10);
fit1 = initial_size_batch_fit(1);


errorbar(initial_size,computation_time,computation_time_std,'.','LineWidth',2,'MarkerSize',10,'Color','b');
plot(initial_size,computation_time_fit,'b-','LineWidth',2,'MarkerSize',10);
fit2 = initial_size_fit(1);

xlim([0,16000000]);

set(gca,'XScale','log','YScale','log','FontSize',fontsz);
xlabel("Number of Cells",'FontSize',fontsz);
ylabel("Collision Computation Time (s)",'FontSize',fontsz);
legend('Serial',['Fit (Slope: ', num2str(fit0,'%3.2f'), ')'],'Batch',['Fit (Slope: ', num2str(fit1,'%3.2f'), ')'],'Mini Batch',['Fit (Slope: ', num2str(fit2,'%3.2f'), ')'],'FontSize',fontsz-2,'Location','SouthEast','box','off');
hold off
drawnow;
saveas(gcf,'collisiontime','fig');
saveas(gcf,'collisiontime','png');




