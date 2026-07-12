% 绘制扰动量x1的截面演化
clear;
mx=256;
mz=mx;
my=32;
aa=0.45;

all_files = dir();
dir_flags = [all_files.isdir] & ~strcmp({all_files.name}, '.') & ~strcmp({all_files.name}, '..');
sub_folders = all_files(dir_flags);
plot_case = {sub_folders.name};
plot_case = strcat(plot_case, filesep);
tk_start=34;
tk_interval=1;
tk_end=100;

figure('visible','on');
colormap(jet);
set(gcf,'Position',[200 200 1000 800]);

for i=1:length(plot_case)
    xx=load([char(plot_case(i)),'/gridxx.dat'])*aa;
    zz=load([char(plot_case(i)),'/gridzz.dat'])*aa;
    % yy=load([char(plot_case(i)),'/gridyy.dat']);
    time=load([char(plot_case(i)),'/nstime.dat']);
    t_end=time(:,4);
    % xmin=min(xx);
    % xmax=max(xx);
    % zmin=min(zz);
    % zmax=max(zz);
    % 画框
    xmin=1.4;
    xmax=2.3;
    zmin=-0.9;
    zmax=0.9;

    [xz_xx,xz_zz]=meshgrid(xx,zz);
    WEIGHT_MTX=ones(mx,mz)';
    xs=load([char(plot_case(i)),'/xxs.dat']);
    zs=load([char(plot_case(i)),'/zzs.dat']);
    xb=xs(:,1)*aa;
    zb=zs(:,1)*aa;
    xss=xb;
    zss=zb;
    xb=[];
    zb=[];
    tsxz=load([char(plot_case(i)),'/gridts.dat']);
    tsxz=reshape(tsxz,mx,mz)';
    dir='./x12d';
    if ~exist(dir, 'dir')
        mkdir(dir);
    end
    for nst=tk_start:tk_interval:tk_end
        disp(nst);
        % plot xd
        str_nst=num2str(nst,'%.3d');
        t=t_end(nst+1);
        str_time=num2str(t);
        w=load([char(plot_case(i)),'/x12d',str_nst]);
        % xz_rho=reshape(w(:,1),mx,mz)';
         xz_p=reshape(w(:,2),mx,mz)';
        % xz_vx=reshape(w(:,3),mx,mz)';
        % xz_vy=reshape(w(:,4),mx,mz)';
        % xz_vz=reshape(w(:,5),mx,mz)';
        % xz_bx=reshape(w(:,6),mx,mz)';
        % xz_by=reshape(w(:,7),mx,mz)';
        % xz_bz=reshape(w(:,8),mx,mz)';
        % xz_br=reshape(w(:,9),mx,mz)';
        % xz_bp=reshape(w(:,10),mx,mz)';
%         xz_vr=reshape(w(:,11),mx,mz)';
        % xz_vp=reshape(w(:,12),mx,mz)';
        % xz_vs=reshape(w(:,13),mx,mz)';

        % contourf(xz_xx,xz_zz,WEIGHT_MTX.*xz_rho,50,'LineStyle','none');fname=['rho' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'-','linewidth',2);hold on;plot(xss,zss,'--');set(gca,'FontSize',30,'FontName','Times New Roman','linewidth',4);hold off;print('-dpng', [dir,'/',fname]);
         contourf(xz_xx,xz_zz,WEIGHT_MTX.*xz_p,50,'LineStyle','none');fname=['p' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'-','linewidth',2);hold on;plot(xss,zss,'--');set(gca,'FontSize',30,'FontName','Times New Roman','linewidth',4);hold off;print('-dpng', [dir,'/',fname]);
        % contourf(xz_xx,xz_zz,WEIGHT_MTX.*xz_vy,50,'LineStyle','none');fname=['vy' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'-','linewidth',2);hold on;plot(xss,zss,'--');set(gca,'FontSize',30,'FontName','Times New Roman','linewidth',4);hold off;print('-dpng', [dir,'/',fname]);
        % contourf(xz_xx,xz_zz,WEIGHT_MTX.*xz_vx,50,'LineStyle','none');fname=['vx' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'-','linewidth',2);hold on;plot(xss,zss,'--');set(gca,'FontSize',30,'FontName','Times New Roman','linewidth',4);hold off;print('-dpng', [dir,'/',fname]);
        % contourf(xz_xx,xz_zz,WEIGHT_MTX.*xz_vz,50,'LineStyle','none');fname=['vz' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'-','linewidth',2);hold on;plot(xss,zss,'--');set(gca,'FontSize',30,'FontName','Times New Roman','linewidth',4);hold off;print('-dpng', [dir,'/',fname]);
%         contourf(xz_xx,xz_zz,WEIGHT_MTX.*xz_vr,50,'LineStyle','none');fname=['vr' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'-','linewidth',2);hold on;plot(xss,zss,'--');set(gca,'FontSize',30,'FontName','Times New Roman','linewidth',4);hold off;print('-dpng', [dir,'/',fname]);
        % contourf(xz_xx,xz_zz,WEIGHT_MTX.*xz_vp,50,'LineStyle','none');fname=['vp' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'-','linewidth',2);hold on;plot(xss,zss,'--');set(gca,'FontSize',30,'FontName','Times New Roman','linewidth',4);hold off;print('-dpng', [dir,'/',fname]);
        % contourf(xz_xx,xz_zz,WEIGHT_MTX.*xz_by,50,'LineStyle','none');fname=['by' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'-','linewidth',2);hold on;plot(xss,zss,'--');set(gca,'FontSize',30,'FontName','Times New Roman','linewidth',4);hold off;print('-dpng', [dir,'/',fname]);
        % contourf(xz_xx,xz_zz,WEIGHT_MTX.*xz_bx,50,'LineStyle','none');fname=['bx' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'-','linewidth',2);hold on;plot(xss,zss,'--');set(gca,'FontSize',30,'FontName','Times New Roman','linewidth',4);hold off;print('-dpng', [dir,'/',fname]);
        % contourf(xz_xx,xz_zz,WEIGHT_MTX.*xz_bz,50,'LineStyle','none');fname=['bz' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'-','linewidth',2);hold on;plot(xss,zss,'--');set(gca,'FontSize',30,'FontName','Times New Roman','linewidth',4);hold off;print('-dpng', [dir,'/',fname]);
        % contourf(xz_xx,xz_zz,WEIGHT_MTX.*xz_br,50,'LineStyle','none');fname=['br' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'-','linewidth',2);hold on;plot(xss,zss,'--');set(gca,'FontSize',30,'FontName','Times New Roman','linewidth',4);hold off;print('-dpng', [dir,'/',fname]);
        % contourf(xz_xx,xz_zz,WEIGHT_MTX.*xz_bp,50,'LineStyle','none');fname=['bp' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'-','linewidth',2);hold on;plot(xss,zss,'--');set(gca,'FontSize',30,'FontName','Times New Roman','linewidth',4);hold off;print('-dpng', [dir,'/',fname]);
        % contourf(xz_xx,xz_zz,WEIGHT_MTX.*xz_vs,50,'LineStyle','none');fname=['vs' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'-','linewidth',2);hold on;plot(xss,zss,'--');set(gca,'FontSize',30,'FontName','Times New Roman','linewidth',4);hold off;print('-dpng', [dir,'/',fname]);

        % plot_xce

        % w=load([char(plot_case(i)),'/xce12d',str_nst]);
        % xz_cx=reshape(w(:,1),mx,mz)';
        % xz_cy=reshape(w(:,2),mx,mz)';
        % xz_cz=reshape(w(:,3),mx,mz)';
        % xz_ex=reshape(w(:,4),mx,mz)';
        % xz_ey=reshape(w(:,5),mx,mz)';
        % xz_ez=reshape(w(:,6),mx,mz)';

        % contourf(xz_xx,xz_zz,WEIGHT_MTX.*xz_cx,50,'LineStyle','none');fname=['Jx' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'-','linewidth',2);hold on;plot(xss,zss,'--');set(gca,'FontSize',30,'FontName','Times New Roman','linewidth',4);hold off;print('-dpng', [dir,'/',fname]);
        % contourf(xz_xx,xz_zz,WEIGHT_MTX.*xz_cy,50,'LineStyle','none');fname=['Jy' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'-','linewidth',2);hold on;plot(xss,zss,'--');set(gca,'FontSize',30,'FontName','Times New Roman','linewidth',4);hold off;print('-dpng', [dir,'/',fname]);
        % contourf(xz_xx,xz_zz,WEIGHT_MTX.*xz_cz,50,'LineStyle','none');fname=['Jz' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'-','linewidth',2);hold on;plot(xss,zss,'--');set(gca,'FontSize',30,'FontName','Times New Roman','linewidth',4);hold off;print('-dpng', [dir,'/',fname]);
        % contourf(xz_xx,xz_zz,WEIGHT_MTX.*xz_ex,50,'LineStyle','none');fname=['Ex' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'-','linewidth',2);hold on;plot(xss,zss,'--');set(gca,'FontSize',30,'FontName','Times New Roman','linewidth',4);hold off;print('-dpng', [dir,'/',fname]);
        % contourf(xz_xx,xz_zz,WEIGHT_MTX.*xz_ey,50,'LineStyle','none');fname=['Ey' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'-','linewidth',2);hold on;plot(xss,zss,'--');set(gca,'FontSize',30,'FontName','Times New Roman','linewidth',4);hold off;print('-dpng', [dir,'/',fname]);
        % contourf(xz_xx,xz_zz,WEIGHT_MTX.*xz_ez,50,'LineStyle','none');fname=['Ez' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'-','linewidth',2);hold on;plot(xss,zss,'--');set(gca,'FontSize',30,'FontName','Times New Roman','linewidth',4);hold off;print('-dpng', [dir,'/',fname]);

    end
end
