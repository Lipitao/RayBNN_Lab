clear all;
close all;
fontsz=16;


RT3_RAD_vec = readmatrix('RT3_RAD_vec.csv');


input_size_list = [6,8,12,14,16,32,50,75,100,125,162]';

meanY = readmatrix("meanY.csv")';
stdY = readmatrix("stdY.csv")';

RT3_RAD_num=length(RT3_RAD_vec);
input_size_list_num=length(input_size_list);
fold_num = 10;

MAE = zeros(fold_num,RT3_RAD_num);
params = zeros(fold_num,RT3_RAD_num);
timearr = zeros(fold_num,RT3_RAD_num);

for RAD_IDX=1:RT3_RAD_num
    RT3_RAD = RT3_RAD_vec(RAD_IDX);
    for fold=1:fold_num
        time_total = 0.0;
        for input_idx=1:input_size_list_num
            input_size = input_size_list(input_idx);

            filename = strcat('fig3a_info_',string(RT3_RAD),'_',string(input_size),'_',string(fold),'.csv');
            info = readmatrix(filename);
            time_total = time_total + info(1);
    
            filename = strcat('fig3a_test_pred_',string(RT3_RAD),'_',string(input_size),'_',string(fold),'.csv');
            pred = readmatrix(filename);
    
            pred = ( pred.*stdY )  + meanY;
    
            filename = strcat('fig3a_test_act_',string(RT3_RAD),'_',string(input_size),'_',string(fold),'.csv');
            act = readmatrix(filename);
    
            act = ( act.*stdY )  + meanY;
    
            MAE(fold,RAD_IDX)=mean(abs(act-pred),"all");

        end
        timearr(fold,RAD_IDX)  = time_total;
    end
end





fig = figure(1);
clf;





MAE_mean = mean(MAE,1);
MAE_std = std(MAE,0,1);

hold on
yyaxis left
set(gca,'FontSize',fontsz);
ylabel("MAE (m)",'FontSize',fontsz);

errorbar(RT3_RAD_vec,MAE_mean,MAE_std,'LineWidth',2);
yyaxis right

plotmean = mean(timearr,1);
plotstd = std(timearr,0,1);

errorbar(RT3_RAD_vec,plotmean,plotstd,'LineWidth',2);
set(gca,'fontsize',fontsz);
xlim([10, 210]);
ylabel("Time (s)",'fontsize',fontsz);
xlabel("RT-3 Radius (r_n)",'FontSize',fontsz);
legend('MAE','Time','FontSize',fontsz,'location','NorthWest');



hold off

drawnow;
saveas(gcf,'RSSIMAE','fig');
saveas(gcf,'RSSIMAE','png');



