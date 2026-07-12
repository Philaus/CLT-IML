% 绘制不同case的增长率随时间变化
% 还可以绘制能量的变化
clear;
all_files = dir();
dir_flags = [all_files.isdir] & ~strcmp({all_files.name}, '.') & ~strcmp({all_files.name}, '..');
sub_folders = all_files(dir_flags);
plot_case = {sub_folders.name};
% plot_case = strcat(plt_case, filesep);
% 绘图时间范围
min_t=800;
max_t=20000;

% cd 'E:\CLTData\plasmoid-p1\'

for i=1:length(plot_case)
    % energy.dat--时间 磁能 动能 热能 增长率 shape=(num, 5)
    dat_dir=['./',char(plot_case(i)),'/energy.dat'];
    energy_data=load(dat_dir);

    time=energy_data(:,1);
    % E_B=energy_data(:,2);
    E_k=energy_data(:,3);
    % E_thermal=energy_data(:,4);
    rate=energy_data(:,5);
    
    figure;
    title('$\gamma-E_k-t$','Interpreter','latex','FontSize', 16);
    xlabel('$t/t_A$','Interpreter','latex','FontSize', 16);
  
    % 左侧绘制增长率
    yyaxis left;
    plot(time,rate,'LineWidth',1.5);
    ylabel('$\gamma$','Interpreter','latex','FontSize', 16);
    % 右侧绘制Ek
    yyaxis right;
    semilogy(time,E_k,'linewidth',0.8, 'LineStyle','--');
    ylabel('$E_k$','Interpreter','latex','FontSize', 16);
    legend(plot_case, 'Location', 'best');
    xlim([min_t,max_t]);
    grid on;
    hold on;
    legend(plot_case{i}, 'Interpreter', 'none', 'Location', 'best'); 
    xlim([min_t,max_t]);
    grid on;

    set(gcf, 'Position', [100, 100, 800, 600]);
    save_path = fullfile('.', char(plot_case(i)), 'plot_growth.png');
    exportgraphics(gcf, save_path, 'Resolution', 300);
    close(gcf);
end
