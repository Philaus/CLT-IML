% clear;
xzero=5;
mx=256;
mz=256;
my=32;
ms=5;
mr=2.60;
mt=326;
aa=0.94;
xx=load('gridxx.dat');
zz=load('gridzz.dat');
yy=load('gridyy.dat');
time=load('nstt.dat');
xx=xx*aa;
zz=zz*aa;
xmin=min(xx);
xmax=max(xx);
zmin=min(zz);
zmax=max(zz);
xs=load('xxs');
zs=load('zzs');
xb=xs(:,1);
zb=zs(:,1);
xb=xb*aa;
zb=zb*aa;
[xz_xx,xz_zz]=meshgrid(xx,zz);
%xint=load('xint.dat');

% xzint_rho=reshape(xint(:,1),mx,mz)';
% xzint_p=reshape(xint(:,2),mx,mz)';
% xzint_vx=reshape(xint(:,3),mx,mz)';
% xzint_vy=reshape(xint(:,4),mx,mz)';
% xzint_vz=reshape(xint(:,5),mx,mz)';
% xzint_bx=reshape(xint(:,6),mx,mz)';
% xzint_by=reshape(xint(:,7),mx,mz)';
% xzint_bz=reshape(xint(:,8),mx,mz)';
% xzint_br=reshape(xint(:,9),mx,mz)';
% xzint_bp=reshape(xint(:,10),mx,mz)';
% cint=load('cint.dat');
% xzint_cx=reshape(cint(:,1),mx,mz)';
% xzint_cy=reshape(cint(:,2),mx,mz)';
% xzint_cz=reshape(cint(:,3),mx,mz)';

% fint=load('fint.dat');
% fx=reshape(fint(:,1),mx,mz)';
% fy=reshape(fint(:,2),mx,mz)';
% fz=reshape(fint(:,3),mx,mz)';
% eint=load('eta.dat');
% eta=reshape(eint(:,1),mx,mz)';
% etax=reshape(eint(:,2),mx,mz)';
% etaz=reshape(eint(:,3),mx,mz)';

% xz_rr=reshape(rr,mx,mz);
% xz_th=reshape(th,mx,mz);
[xz_xx,xz_zz]=meshgrid(xx,zz);
% % xz_xx=xz_xx';
% % xz_zz=xz_zz';
% [rt_at,rt_ar]=meshgrid(at,ar);
% [rt_xx,rt_zz]=pol2cart(rt_at,rt_ar);
% xs=reshape(xz_xx,mx*mz,1);
% zs=reshape(xz_zz,mx*mz,1);
% scatter(xs,zs,1);hold on;
% scatter(xxb,zzb,5,'r');hold on;
% alpha=0:pi/160:2*pi; % Angle [0, 2*pi].
% R=min(rrb); % Radius.
% x=xzero+R*cos(alpha); 
% y=R*sin(alpha); 
% plot(x,y,'-');hold on;
% hold off;
mkdir png2;
ndst=1;
nstartime=0;
nstart=0;
colormap(jet);
p0=load('pt0.dat');
p0=reshape(p0(:,1),mx,mz)';
for nst=nstart:ndst:100

str_nst=num2str(nst,'%.3d');
str_nst1=num2str(nst+1,'%.3d');
str_time=num2str(time((nst-nstartime)/ndst+1,2));
w=load(['x12d' str_nst ]);
xz_p=reshape(w(:,2),mx,mz)';
pt=p0+xz_p;
pt(find(pt<0))=0;

 [C,h]=contourf(xz_xx,xz_zz,pt,50);figure(gcf); set(h, 'LineStyle', 'none');fname=['pt' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);caxis([0,0.0073]);hold on;plot(xb,zb,'--');hold off;print('-dpng', ['png2\' fname]);
h=plot(xx',(pt(128,:)),'linewidth',4);axis([min(xx),max(xx),0,0.0075]);
fname=['ptx' str_nst];title([ fname '   t=' str_time ]);xlabel('R');print('-dpng', ['png2\' fname]);

axis square
h=plot(zz',pt(:,148),'linewidth',4);
axis([min(zz),max(zz),0,0.0075]);
axis square
 fname=['ptz' str_nst ];title([ fname '   t=' str_time ]);xlabel('Z');print('-dpng', ['png2\' fname]);
% [C,h]=contourf(xz_xx,xz_zz,pt,50);figure(gcf); set(h, 'LineStyle', 'none');fname=['pt' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');caxis([0,0.0070]);colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'--');hold off;print('-dpng', ['png2\' fname]);
%  [C,h]=contourf(xz_xx,xz_zz,xz_vy,50);figure(gcf); set(h, 'LineStyle', 'none');fname=['vy' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'--');hold off;print('-dpng', ['png1\' fname]);
%  [C,h]=contourf(xz_xx,xz_zz,xz_vx,50);figure(gcf); set(h, 'LineStyle', 'none');fname=['vx' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'--');hold off;print('-dpng', ['png1\' fname]);
%  [C,h]=contourf(xz_xx,xz_zz,xz_vz,50);figure(gcf); set(h, 'LineStyle', 'none');fname=['vz' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'--');hold off;print('-dpng', ['png1\' fname]);
%  [C,h]=contourf(xz_xx,xz_zz,xz_vr,50);figure(gcf); set(h, 'LineStyle', 'none');fname=['vr' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'--');hold off;print('-dpng', ['png1\' fname]);
% [C,h]=contourf(xz_xx,xz_zz,xz_vp,50);figure(gcf); set(h, 'LineStyle', 'none');fname=['vp' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'--');hold off;print('-dpng', ['png1\' fname]);
%  [C,h]=contourf(xz_xx,xz_zz,xz_by,50);figure(gcf); set(h, 'LineStyle', 'none');fname=['by' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'--');hold off;print('-dpng', ['png1\' fname]);
%  [C,h]=contourf(xz_xx,xz_zz,xz_bx,50);figure(gcf); set(h, 'LineStyle', 'none');fname=['bx' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'--');hold off;print('-dpng', ['png1\' fname]);
%  [C,h]=contourf(xz_xx,xz_zz,xz_bz,50);figure(gcf); set(h, 'LineStyle', 'none');fname=['bz' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'--');hold off;print('-dpng', ['png1\' fname]);
%  [C,h]=contourf(xz_xx,xz_zz,xz_br,50);figure(gcf); set(h, 'LineStyle', 'none');fname=['br' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'--');hold off;print('-dpng', ['png1\' fname]);
%  [C,h]=contourf(xz_xx,xz_zz,xz_bp,50);figure(gcf); set(h, 'LineStyle', 'none');fname=['bp' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'--');hold off;print('-dpng', ['png1\' fname]);
%  [C,h]=contourf(xz_xx,xz_zz,xz_cx,50);figure(gcf); set(h, 'LineStyle', 'none');fname=['cx' str_nst ];title(fname);colorbar;print('-dpng', ['png1\' fname]);
%  [C,h]=contourf(xz_xx,xz_zz,xz_cy,50);figure(gcf); set(h, 'LineStyle', 'none');fname=['cy' str_nst ];title(fname);colorbar;print('-dpng', ['png1\' fname]);
% [C,h]=contourf(xz_xx,xz_zz,xz_cz,50);figure(gcf); set(h, 'LineStyle', 'none');fname=['cz' str_nst ];title(fname);colorbar;print('-dpng', ['png1\' fname]);
% [C,h]=contourf(xz_xx,xz_zz,xz_ex,50);figure(gcf); set(h, 'LineStyle', 'none');fname=['ex' str_nst ];title(fname);colorbar;print('-dpng', ['png1\' fname]);
% [C,h]=contourf(xz_xx,xz_zz,xz_ey,50);figure(gcf); set(h, 'LineStyle', 'none');fname=['ey' str_nst ];title(fname);colorbar;print('-dpng', ['png1\' fname]);
% [C,h]=contourf(xz_xx,xz_zz,xz_ez,50);figure(gcf); set(h, 'LineStyle', 'none');fname=['ez' str_nst ];title(fname);colorbar;print('-dpng', ['png1\' fname]);
% 
%   [C,h]=contourf(xz_xx,xz_zz,xz_vs,50);figure(gcf); set(h, 'LineStyle', 'none');fname=['vs' str_nst ];title([ fname '   t=' str_time ]);xlabel('R');ylabel('Z');colorbar;axis('equal',[xmin xmax zmin zmax ]);hold on;plot(xb,zb,'--');hold off;print('-dpng', ['png1\' fname]);

end
% 
% for jr=1:mr
%     for jt=1:mt
%         xl(jr,jt)=xzero+ar(jr)*cos(at(jt));
%         zl(jr,jt)=ar(jr)*sin(at(jt));
%     end
% end
