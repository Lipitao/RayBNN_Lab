clear all;
close all;
fontsz=16;






input_size = 162;

meanY = readmatrix("meanY.csv")';
stdY = readmatrix("stdY.csv")';

fold_num=10;

error_list = [];

for fold=1:fold_num

    filename = strcat('info_',string(fold),'_',string(input_size),'.csv');
    info = readmatrix(filename);


    filename = strcat('test_pred_',string(fold),'_',string(input_size),'.csv');
    pred = readmatrix(filename);

    pred = ( pred.*stdY )  + meanY;

    filename = strcat('test_act_',string(fold),'_',string(input_size),'.csv');
    act = readmatrix(filename);

    act = ( act.*stdY )  + meanY;

    diff = act - pred;
    
    diff = (diff) .^2;
    diff = sum(diff, 2);
    
    diff = sqrt(diff);

    error_list = cat(1,error_list,diff);

end



fig = figure(1);
clf;




hold on


[p,x] = hist(error_list,100); 


plot(x,p/sum(p),'LineWidth',2,'MarkerSize',10);


xlim([0,5]);
%ylim([0.84,2.4]);
neural_network_list = {'\color{red}\bf RayBNN','CNN','GCN2','LSTM','MLP','GCN2LSTM','BILSTM'};
set(gca,'Box','on','FontSize',fontsz);
legend(neural_network_list,'Location','northeast','FontSize',fontsz,'box','off');
xlabel('Error (m)','FontSize',fontsz);
ylabel('Probability','FontSize',fontsz);
hold off
drawnow;

saveas(gcf,'RSSIPDF','fig');
saveas(gcf,'RSSIPDF','png');





