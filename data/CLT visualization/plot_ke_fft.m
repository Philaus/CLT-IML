 energy_kyn = load('case/energy_kyn.dat');
[T, n_max] = size(energy_kyn);
n_max = n_max - 8;

color_name = {'Blue';'Red';'Black';'Green';'Cyan';'Yellow';'Magenta';};
toroidal_name   = cell(n_max+1,1);

for n=0:n_max
    if(n<7) 
        semilogy(energy_kyn(:,1),energy_kyn(:,n+2),'Color',color_name{n+1},'Linewidth',1);
    else
        semilogy(energy_kyn(:,1),energy_kyn(:,n+2),'Color',[rand(1) rand(1) rand(1)],'Linewidth',1);
    end
    toroidal_name{n+1} = ['n=',num2str(n)];
    hold on
end
legend(toroidal_name,'Fontsize',10,'Location','northeastoutside','FontSize',16);
xlabel('time','Interpreter','latex','FontSize',36)
ylabel('Kinetic Energy','Interpreter','latex','FontSize',36)
set(gca,'Fontsize',20)
set(gcf, 'Position', [100, 100, 800, 600]);
xlim([200,6000]);
exportgraphics(gcf, 'ke_fft.png', 'Resolution', 300);