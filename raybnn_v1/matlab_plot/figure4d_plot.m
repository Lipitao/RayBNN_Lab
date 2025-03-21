clear all;
close all;
fontsz=16;

input_size_list = [6,8,12,14,16,32,50,75,100,125,162]';

meanY = readmatrix("meanY.csv")';
stdY = readmatrix("stdY.csv")';

fold_num=10;
input_size_num = length(input_size_list);

MAE = zeros(input_size_num,fold_num);
params = zeros(input_size_num,fold_num);
timearr = zeros(input_size_num,fold_num);

for fold=1:fold_num
    for i=1:input_size_num
        input_size = input_size_list(i);

        filename = strcat('info_',string(fold),'_',string(input_size),'.csv');
        info = readmatrix(filename);

        filename = strcat('test_pred_',string(fold),'_',string(input_size),'.csv');
        pred = readmatrix(filename);

        pred = ( pred.*stdY )  + meanY;

        filename = strcat('test_act_',string(fold),'_',string(input_size),'.csv');
        act = readmatrix(filename);

        act = ( act.*stdY )  + meanY;

        MAE(i,fold)=mean(abs(act-pred),"all");


    end
end





fig = figure(1);
clf;


hold on


MAE_mean = mean(MAE,2);
MAE_std = std(MAE,0,2);

errorbar(input_size_list,MAE_mean,MAE_std,'LineWidth',2);

xlim([0,170]);
ylim([0.84,2.4]);
neural_network_list = {'\color{red}\bf RayBNN','CNN','GCN2','LSTM','MLP','GCN2LSTM','BILSTM'};
legend(neural_network_list ,'Location','NorthEast','box','off','FontSize',fontsz,'NumColumns',2);
xlabel('Num. APs','FontSize',fontsz);
ylabel('Mean Absolute Error (m)','FontSize',fontsz);
hold off
set(gca,'FontSize',fontsz,'box','on');

drawnow;
saveas(gcf,'RSSIMAE','fig');
saveas(gcf,'RSSIMAE','png');



