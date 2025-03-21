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



cdf = error_list;


[f,x] = ecdf(cdf); 

xsize = uint32(size(x,1)/30);
x = x(1:xsize:end);
f = f(1:xsize:end);

plot(x,f,'LineWidth',2,'MarkerSize',10);

xlim([0,5]);
%ylim([0.84,2.4]);
set(gca,'box','on','FontSize',fontsz);


neural_network_list = {'\color{red}\bf RayBNN','CNN','GCN2','LSTM','MLP','GCN2LSTM','BILSTM'};

%legend(neural_network_list);
legend(neural_network_list,'Location','southeast','FontSize',fontsz,'box','off');
xlabel('Error (m)','FontSize',fontsz);
ylabel('Cumulative Probability','FontSize',fontsz);
hold off

drawnow;
saveas(gcf,'RSSICDF','fig');
saveas(gcf,'RSSICDF','png');



