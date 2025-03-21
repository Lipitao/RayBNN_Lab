clear all
close all
fontsz=16;

hold on
neuron_size = 700;

WAdj = zeros(neuron_size,neuron_size);
rn = 0.1;

for i=0:10

    model_filename = strcat("./sparsenetwork_",string(i),".csv");


    [WValues,WRowIdxCSR,WColIdx,H,A,B,C,D,E,glia_pos,neuron_pos,neuron_idx,net_data] = readNetworkFile(model_filename);

    neuron_size = net_data(1);
    input_size = net_data(2);
    output_size = net_data(3);

    active_size = size(neuron_idx,1);


    WRowIdxCOO = CSRtoCOO(uint32(WRowIdxCSR));


    WRowIdx = uint32(WRowIdxCOO)+1;
    WColIdx = uint32(WColIdx)+1;
    neuron_idx = uint32(neuron_idx)+1;






    active_size = size(neuron_pos,1)/3;

    neuron_pos = reshape(neuron_pos,[active_size ,3]);

    maxdist = round(sqrt(max(sum(neuron_pos .^ 2,2 ))))/rn;


    WValues_mean=mean(WValues);

    WValues_std=std(WValues);





    % max_weight = 2.0*mean(abs(WValues));
    max_weight = WValues_mean+2*WValues_std;
    min_weight=WValues_mean-2*WValues_std;

    % WValues(WValues> max_weight)=max_weight;
    % WValues(WValues< -max_weight)=-max_weight;
    colormap('jet');
    % colormap(redblue)

    cm=colormap;


    randmaxtrix = rand(size(WColIdx));
    index =  (randmaxtrix > 0.9);

    WColIdx = WColIdx(index);
    WRowIdx = WRowIdx(index);
    WValues = WValues(index);


    Zdir = ones(size(WColIdx))*i;


    colors = ones(length(WValues),3);
    for qq = 1:length(colors)
        % if WValues(qq) > 0.0
        %     valz = 1-(abs(WValues(qq))/max_weight);
        %     colors(qq,:) = [1 valz valz];
        % end
        % if WValues(qq) < 0.0
        %     valz = 1-(abs(WValues(qq))/max_weight);
        %     colors(qq,:) = [valz valz 1];
        % end
        [~,colorID] = min(abs(WValues(qq) - linspace(min_weight,max_weight,size(cm,1))));
        colors(qq,:)=cm(colorID, :);
    end


    scatter3(WRowIdx,Zdir,WColIdx,10,colors,'filled');
    %scatter3(WRowIdx,Zdir,WColIdx,'filled');

end
xlabel("Row",'FontSize',fontsz,'rotation',-25,'VerticalAlignment','top','HorizontalAlignment','right');
ylabel("Iteration",'FontSize',fontsz,'rotation',25,'VerticalAlignment','top','HorizontalAlignment','left');
zlabel("Column",'FontSize',fontsz);
% set(gca, 'clim', [min_weight  max_weight]);

set(gca, 'clim', [min_weight  max_weight],'FontSize',fontsz);
c=colorbar('FontSize',fontsz);

ylabel(c,'Weights','FontSize',fontsz);

view([45,-45,45]);
grid on;

hold off
drawnow;

% for i = 1:3
exportgraphics(gcf,'WAdj.png','Resolution',600);
savefig('WAdj.fig');
%     pause(1);
% end
