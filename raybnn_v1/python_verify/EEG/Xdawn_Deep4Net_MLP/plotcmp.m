


clear all
close all


optim_raybnn = readmatrix("optim_raybnn.txt");
optim_MLP = readmatrix("optim_MLP.txt");

hold on
plot(optim_raybnn(:,7) ,optim_raybnn(:,5), 'Linewidth',2);
plot(optim_MLP(:,7) ,optim_MLP(:,5), 'Linewidth',2);


xlim([0 1600000])
ax=gca;
ax.FontSize = 16;

xlabel('Trainable Parameters','FontSize',16);
ylabel('ROC AUC','FontSize',16);
legend('Xdawn-Deep4Ne-RayBNN','Xdawn-Deep4Ne-MLP with Dropout','FontSize',16,'Location','SouthEast','box','off');
hold off
exportgraphics(gcf,'RayBNNMLPcmp.png','Resolution',600);
saveas(gcf,'RayBNNMLPcmp','fig');






