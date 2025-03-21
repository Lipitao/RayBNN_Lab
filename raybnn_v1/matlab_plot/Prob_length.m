% clear all;
% close all;
function [r_array,p_array]=Prob_length(density_list)
rs=739.81;
% density_list=[ 0.0050    0.0100    0.0200    0.0400    0.0500    0.0600    0.0800]*1e-2;
p_array=cell(length(density_list),1);
rm=40;
r_array=linspace(0,2*rm,100);
r_array_norm=r_array/rm;
figure(1);clf;

for count=1:length(density_list)
    N_n=(density_list(count)/2)*(4*pi/3)*rm^3;
    N_g=N_n;
    eta_n=density_list(count)/2;
    eta=density_list(count);
    p_array{count}=(r_array.^2.*(1-pi*eta*r_array))...
        .*((r_array-2*rm).^2).*(r_array+4*rm);
     p_array{count}= p_array{count};
    K_fac=sum(p_array{count})*diff(r_array(1:2))
    p_array{count}=p_array{count}/K_fac;
    plot3(eta*ones(size(r_array)),r_array,p_array{count},'linewidth',2); hold on;

end
% set(gca,'XScale','log');
xlabel('Density (r_n^{-3})','FontSize',14,'rotation',19,'VerticalAlignment','top','HorizontalAlignment','left');
ylabel('L_n/r_n','FontSize',14);
zlabel('Probability','FontSize',14);
ylim([0,2*rm])
zlim([0,1.2*max(p_array{end})]);
set(gca,'FontSize',12);
grid on;
drawnow;
savefig(gcf,"length_prob_theory.fig");
saveas(gcf,'length_prob_theory','png');
% figure(2);clf;
% semilogy(r_array,(1-pi*eta*r_array));
%
% syms eta_n r_array rm rn eta
% p_array=(pi*eta_n/(4*rm^3))*(r_array*(1-pi*eta*r_array))...
%     *((((rm-r_array)^3)*(3*rm+17*r_array))+rm^2*(6*r_array^2+8*rm*r_array-3*rm^2));
% diff(p_array,r_array)
% p_array=(pi*eta_n/(4*rm^3))*r_array...
%     *((((rm-r_array)^3)*(3*rm+17*r_array))+rm^2*(6*r_array^2+8*rm*r_array-3*rm^2));
% diff(p_array,r_array)