clear all
close all
fontsz=16;


inx = -5:0.005:5;

model_filename = './sparsenetwork_0.csv';


[WValues,WRowIdxCSR,WColIdx,H,A,B,C,D,E,glia_pos,neuron_pos,neuron_idx,net_data] = readNetworkFile(model_filename);

neuron_size = net_data(1);
input_size = net_data(2);
output_size = net_data(3);

active_size = size(neuron_idx,1);


WRowIdxCOO = CSRtoCOO(uint32(WRowIdxCSR));


WRowIdx = uint32(WRowIdxCOO)+1;
WColIdx = uint32(WColIdx)+1;
neuron_idx = uint32(neuron_idx)+1;
UAF_mat=zeros(active_size-input_size+1,length(inx));

Wi = sparse(WRowIdx,WColIdx,WValues,neuron_size,neuron_size);
Wi = full(Wi);

G = digraph(Wi);
weight=WValues;

IND = indegree(G);
Q = outdegree(G);
index = find(Q == 0);
index = index(input_size+1:end);

index = find(Q == 0 & IND == 0);





figure
%nexttile
hold on

xlabel("f_{UAF} Input",'FontSize',fontsz);
ylabel("f_{UAF} Output",'FontSize',fontsz);

%title('Activation Functions');
Ai = A;
Bi = B;
Ci = C;
Di = D;
Ei = E;

Ai(index) = [];
Bi(index) = [];
Ci(index) = [];
Di(index) = [];
Ei(index) = [];
figobj = [];
for j=input_size:active_size
    y = UAF(inx,Ai(j),Bi(j),Ci(j),Di(j),Ei(j));
%     figobj = [  figobj ; plot(inx,y) ];
    UAF_mat(j-input_size+1,:)=y;
end
% hold off



% 
% 
% for i = 1:3
%     exportgraphics(gcf,'RSSIUAF.pdf','Resolution',600);
%     savefig('RSSIUAF.fig');
%     pause(1);
% end



figure(1);clf;
for count=input_size:active_size
    plot3(inx,count*ones(1,length(inx)),UAF_mat(count-input_size+1,:),'LineWidth',2);
    hold on;
end
xlabel("UAF Input",'FontSize',fontsz,'rotation',18,'VerticalAlignment','top','HorizontalAlignment','left');
ylabel("Neuron",'FontSize',fontsz,'rotation',-25,'VerticalAlignment','top','HorizontalAlignment','right');
zlabel("UAF Output",'FontSize',fontsz);
set(gca,'FontSize',fontsz);
grid on;
drawnow;
saveas(gcf,'RSSIUAF3D','fig');
saveas(gcf,'RSSIUAF3D','png');



