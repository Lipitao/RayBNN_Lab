clear all;
close all;
fontsz=16;


input_size = [6, 8, 12, 14,  16, 32, 50, 75, 100, 125, 162];

zeroelems = zeros(11,1);
nonzero = zeros(11,1);

sparsity = zeros(11,1);
for i = 0:10

    filename = strcat('sparsenetwork_', string(i),'.csv');
    
    %Read Neural Network file
    [WValues,WRowIdxCSR,WColIdx,H,A,B,C,D,E,glia_pos,neuron_pos,neuron_idx,net_data] = readNetworkFile(filename);

    Wsize = length(WColIdx);

    active_size = size(neuron_idx,1);

    zeroelems(i+1) = active_size.^ 2;
    nonzero(i+1) = Wsize;

    sparsity(i+1) = nonzero(i+1) ./ zeroelems(i+1);
end



figure(1);clf;
yyaxis left;
plot(input_size,nonzero,'bo-.','LineWidth',2,'MarkerSize',10);
hold on;
plot(input_size,zeroelems,'ks--','LineWidth',2,'MarkerSize',10);
set(gca,'XScale','log','YScale','log','YColor','k','FontSize',fontsz);

xlabel('Num. APs','FontSize',fontsz);
ylabel('Counts','FontSize',fontsz);

yyaxis right;
plot(input_size,100*sparsity,'LineStyle','-','Color','r','LineWidth',2, 'Marker','+','MarkerSize',10);
legend('Non-zeros','Total','Sparsity','FontSize',fontsz,'Location','South','box','off');
set(gca,'YColor','k','FontSize',fontsz);
ylabel('Sparsity (%)','FontSize',fontsz);
ylim([0 100]);
saveas(gcf,'sparsity','fig');
saveas(gcf,'sparsity','png');