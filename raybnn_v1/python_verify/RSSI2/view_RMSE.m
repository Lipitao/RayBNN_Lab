



input_size_list = [6,8,12,14,16,32,50,75,100,125,162]';


neural_network_list = {'BNN','CNN','GCN2','LSTM','MLP'};


hold on

for  i=1:length(neural_network_list)
    
    network = neural_network_list(i);
    network = network{1};

    MAE = readmatrix(join(['./',network,'/RMSE.csv'])    );
    MAE_mean = mean(MAE,2);
    MAE_std = std(MAE,0,2);
    
    errorbar(input_size_list,MAE_mean,MAE_std);
end

legend(neural_network_list);
xlabel('Number of APs')
ylabel('Root Mean Square Error (m)')
hold off





