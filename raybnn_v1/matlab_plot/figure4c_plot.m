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

        timearr(i,fold)  = info(1);
        params(i,fold)  = info(2);

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


timearr = cumsum(timearr);
timearr_mean = mean(timearr,2);
timearr_std = std(timearr,0,2);

errorbar(input_size_list,timearr_mean,timearr_std,'LineWidth',2,'MarkerSize',10);


xlim([0,170]);
ylim([1,3000]);


set(gca,'YScale','log','FontSize',fontsz,'box','on');
neural_network_list = {'\color{red}\bf RayBNN','CNN','GCN2','LSTM','MLP','GCN2LSTM','BILSTM'};

legend(neural_network_list,'Location','southeast','box','off','FontSize',fontsz,'NumColumns',2);
xlabel('Num. APs','FontSize',fontsz)
ylabel('Cumulative Training Time (s)','FontSize',fontsz)
hold off


saveas(gcf,'RSSITimeTotal','fig');
saveas(gcf,'RSSITimeTotal','png');



