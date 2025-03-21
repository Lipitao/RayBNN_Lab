clear all
close all
fontsz=16;


acc_deep4net = readmatrix('acc_deep4net.txt');
acc_CSP_LDA = readmatrix('acc_CSP_LDA.txt');
acc_CSP_LR = readmatrix('acc_CSP_LR.txt');
acc_deep4net_raybnn = readmatrix('acc_deep4net_raybnn.txt');
acc_ensemble = readmatrix('acc_ensemble.txt');
acc_xdawn_LR = readmatrix('acc_xdawn_LR.txt');
acc_xdawn_MDM = readmatrix('acc_xdawn_MDM.txt');
acc_xdawn_deep4net_mlp = readmatrix('acc_xdawn_deep4net_mlp.txt');


figure(1);clf;
hold on
plot(acc_CSP_LDA(:,1), '--' , 'linewidth',2,'markersize',8)
plot(acc_xdawn_LR(:,1), 'square',  'linewidth',2,'markersize',8)
plot(acc_deep4net_raybnn(:,1),  'b' , 'linewidth',2,'markersize',8)

legendstr={'CSP-LDA',
 'Xdawn-LR',
'Deep4Net-RayBNN',
'CSP-LR', 
'Deep4Net', 
'Deep4Net-Xdawn-RayBNN', 
'Xdawn-MDM',
'Xdawn-Deep4Net-MLP',
 };


plot(acc_CSP_LR(:,1), '--',  'linewidth',2,'markersize',8)
plot(acc_deep4net(:,1), '^',  'linewidth',2,'markersize',8)
plot(acc_ensemble(:,1),  'r' , 'linewidth',2,'markersize',8)


plot(acc_xdawn_MDM(:,1), '--',  'linewidth',2,'markersize',8)
plot(acc_xdawn_deep4net_mlp(:,1), '*',  'linewidth',2,'markersize',8)




xlim([1  54])
ylim([0.3  1.0])
hold off


a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','Times','fontsize',fontsz)



legend(legendstr,'Location','SouthWest','FontSize',fontsz,'Box','off','NumColumns',3);
ylabel('Testing Accuracy','FontSize',fontsz);
xlabel('Dataset Test Index','FontSize',fontsz);
saveas(gcf,'EEG_MI','fig');
exportgraphics(gcf,'EEG_MI.png','Resolution',600);



%diff = acc_ensemble - acc_xdawn_deep4net_mlp;


%[h,p,ci,stats] = ttest(diff)

format shortE 

arr = zeros(10,size(acc_ensemble,2));

for c = 1:size(acc_ensemble,2)
    [h,p] = ttest(acc_ensemble(:,c) , acc_xdawn_deep4net_mlp(:,c), "Tail","right");
    arr(1,c) = p;
end

for c = 1:size(acc_ensemble,2)
    [h,p] = ttest(acc_ensemble(:,c) , acc_deep4net(:,c), "Tail","right");
    arr(2,c) = p;
end

for c = 1:size(acc_ensemble,2)
    [h,p] = ttest(acc_ensemble(:,c) , acc_xdawn_LR(:,c), "Tail","right");
    arr(3,c) = p;
end

for c = 1:size(acc_ensemble,2)
    [h,p] = ttest(acc_ensemble(:,c) , acc_xdawn_MDM(:,c), "Tail","right");
    arr(4,c) = p;
end

for c = 1:size(acc_ensemble,2)
    [h,p] = ttest(acc_ensemble(:,c) , acc_CSP_LR(:,c), "Tail","right");
    arr(5,c) = p;
end


for c = 1:size(acc_ensemble,2)
    [h,p] = ttest(acc_ensemble(:,c) , acc_CSP_LDA(:,c), "Tail","right");
    arr(6,c) = p;
end




