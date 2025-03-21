% Plot Figure 1b Generating an Example Neural Network Sphere 
% 
% Input files
% ./figure1_neural_network.csv    Neural network file containing the cell positions and the weights of the neural connections
%

filename = './figure1_neural_network.csv';

%Read Neural Network file
[WValues,WRowIdxCSR,WColIdx,H,A,B,C,D,E,glia_pos,neuron_pos,neuron_idx,net_data] = readNetworkFile(filename);

%Resize Network to Neural Network Sphere Radius
rs = max(neuron_pos,[],"all")*1.2;
neuron_pos=neuron_pos/rs;
glia_pos=glia_pos/rs;

%Get Network Parameters
neuron_size = net_data(1);
input_size = net_data(2);
output_size = net_data(3);

active_size = size(neuron_idx,1);

WRowIdxCOO = CSRtoCOO(uint32(WRowIdxCSR));

WRowIdxCOO = uint32(WRowIdxCOO)+1;
WColIdx = uint32(WColIdx)+1;
neuron_idx = uint32(neuron_idx)+1;




close all


%Reshape Cell positions

neuron_pos = reshape(neuron_pos,[active_size ,3]);
total_neuron_pos = zeros(neuron_size,3);
total_neuron_pos(neuron_idx,:) =  neuron_pos;

glia_pos = reshape(glia_pos,[length(glia_pos)/3 ,3]);
WValues_mean=mean(WValues);
WValues_std=std(WValues);





max_weight = WValues_mean+2*WValues_std;
min_weight=WValues_mean-2*WValues_std;




%Plot Cells in the neural network

hold on

colormap('jet');

%Plot the individual cells

cm=colormap;
scatter3(glia_pos(:,1),glia_pos(:,2),glia_pos(:,3), 150,"filled"  ,"MarkerFaceColor" ,'r');

scatter3(neuron_pos(:,1),neuron_pos(:,2),neuron_pos(:,3), 150,"filled"  ,"MarkerFaceColor" ,'g');
legendstr={'Glial','Neuron'};

for i= 1:20:size(WColIdx,1)

    %Plot the connections between the cells
    start = WColIdx(i);
    finish = WRowIdxCOO(i);
    p1 = total_neuron_pos(start,:);
    p2 = total_neuron_pos(finish,:);


    weight=WValues(i);
    [~,colorID] = min(abs(weight - linspace(min_weight,max_weight,size(cm,1))));
    myColor = cm(colorID, :);

    dp = p2-p1;
    
    quiver3( p1(1),p1(2),p1(3)  ,dp(1),dp(2),dp(3),0  ,'color',myColor,'linewidth',2);
    legendstr{end+1}='';
end
set(gca, 'clim', [min_weight  max_weight]);
c=colorbar('FontSize',12);



ylabel(c,'Weights','FontSize',14);

legend(legendstr,'FontSize',14,'box','off');


grid on
xlabel('X/r_s','fontsize',14,'rotation',25,'VerticalAlignment','top','HorizontalAlignment','left');
ylabel('Y/r_s','FontSize',14,'rotation',-25,'VerticalAlignment','top','HorizontalAlignment','right');
zlabel('Z/r_s','FontSize',14);
% set(gca,'Color','k')
set(gca,'FontSize',12);
view(3)
% hold off

drawnow;

%Save the plot
exportgraphics(gcf,'RSSI3D.png','Resolution',600);



