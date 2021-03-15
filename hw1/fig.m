%% No preprocessing & Gradient Ascent
load('./data/no_pre_Grad_errRate.txt');
load('./data/no_pre_Grad_loss.txt');
load('./data/no_pre_Grad_weight_0.txt');
load('./data/no_pre_Grad_weight_1.txt');
load('./data/no_pre_Grad_weight_2.txt');
load('./data/no_pre_Grad_weights.txt');

iter = size(no_pre_Grad_errRate, 1);
iter_ax = (1:iter)';
Color = {[0 0 0],[0 0.5 1],[0 0 1],[0 1 0],[1 0 1],[1 0 0],[0 1 1],[1 1 0],[0.2 0.7 0.4],[0.7 0.5 0.3]};                          

fig_loss = figure();
plot(iter_ax, no_pre_Grad_loss, 'Color', Color{3}, 'LineStyle', '-', 'linewidth', 2);
xlabel('Iteration');
ylabel('Loss');
set(gca, 'FontSize', 14, 'FontWeight', 'bold');



% the plot operation 

EbN0Vec = (-7:3)';

Marker = {'o','*','s','d','^','v','>','<','p','h'};

for i = 1:size(BER,1)
    % if i==10
        % semilogy(EbN0Vec+11, BER(i,:), 'Color', Color{i}, 'LineStyle', '-', 'linewidth', 2);
    % else
        semilogy(EbN0Vec+11, BER(i,:), 'Color', Color{i}, 'LineStyle', '-', 'Marker', Marker{i}, 'MarkerSize', 6, 'linewidth', 2);
    % end
    hold on
end

grid on
xlabel('SNR in dB');
ylabel('BER');
set(gca, 'FontSize', 14, 'FontWeight', 'bold');
set(gca,'XLim',[EbN0Vec(1,1)+11 EbN0Vec(11,1)+11]);
% set(gca,'XLim',[0 10]);
% set(gca,'FontName','Times New Roman','FontSize',14)设置坐标轴刻度字体名称，大小‘FontWeight�??,’bold�?? 加粗 ‘FontAngle�??,’italic�?? 斜体
% % legend('NS', ... 
% %        'CG', 'GS',...
% %        'PCI', 'Jacobi',...
% %        'Richardson','proposed-1',...
% %        'proposed-2','TMA', 'MMSE',...
% %        'Location','southwest');
% % % h = legend( 'Traditional Richardson', ...
% % %         'Proposed(zero)', ...
% % %         'Proposed($\omega \times y$)', ...
% % %         'Proposed($D^{-1}\times y$)' ...
% % % );
legend( 'MMSE', ...
        'NS', ...
        'CG', ...
        'Jacobi', ...
        'Traditional Richardson', ...
        'Proposed' ...
);
% h=legend('$z=\frac{{x^2}}{{2^2}}+\frac{{y^2}}{{4^2}}$'); %latex分式
% set(h,'Interpreter','latex','Location','Southwest')
% 'Location','southwest'
