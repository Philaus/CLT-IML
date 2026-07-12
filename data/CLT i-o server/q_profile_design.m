%this code is used to design reverse sheared equilibrium
clear;
n=211;
mmax=90;
kmax=90;
jmax=60;
imax=70;

% 搜索基准点
lamuta0=7.48;
r0=0.8;
delta=0.4;
r_delta=0;
A0=1.0;
qc=4.0;

dr0=0.3/jmax;
ddelta=0.35/imax;
dlamuta=5/kmax;
dA0=2/mmax;
r=0:1/n:1;

% 参数指定初始值
r10=0.25;
r20=0.603;

s10=-0.6886;
s20=0.8039;

dis_r0=r20-r10;
delta_s0=s20-s10;

for step_r2=-2:1
    r20=0.603+0.04*step_r2;
    for step_r1=0:3
        r10=0.25+0.05*step_r1;
        for m=1:mmax
            for k=1:kmax
                for i=1:imax
                    for j=1:jmax
                        r01(j)=r0-j*dr0;
                        delta1(i)=delta-i*ddelta;
                        lamuta(k)=lamuta0-k*dlamuta;
                        A01(m)=A0+m*dA0;
                        q=qc*(1+(r/r01(j)).^(2*lamuta(k))).^(1/lamuta(k)).*(1+A01(m)*exp(-((r-r_delta)/delta1(i)).^2))/(1+A01(m));
                        dqdr=diff(q)./diff(r);
                        dqdr=[dqdr dqdr(end)];
                        s=r./q.*dqdr;
                        if(isempty(find(q<2)))
                            r1(i,j,k,m)=NaN;
                            r2(i,j,k,m)=NaN;
                            dis_r(i,j,k,m)=NaN;
                            s1(i,j,k,m)=NaN;
                            s2(i,j,k,m)=NaN;
                            delta_s(i,j,k,m)=NaN;
                            err(i,j,k,m)=NaN;
                        else
                            r1(i,j,k,m)=r(min(find(q<2)));
                            r2(i,j,k,m)=r(max(find(q<2)));
                            dis_r(i,j,k,m)=r2(i,j,k,m)-r1(i,j,k,m);
                            s1(i,j,k,m)=s(min(find(q<2)));
                            s2(i,j,k,m)=s(max(find(q<2)));
                            delta_s(i,j,k,m)=s2(i,j,k,m)-s1(i,j,k,m);
                            err(i,j,k,m)=((r1(i,j,k,m)-r10)/dis_r0).^2+((r2(i,j,k,m)-r20)/dis_r0).^2.+((s1(i,j,k,m)-s10)/delta_s0).^2.+((s2(i,j,k,m)-s20)/delta_s0).^2;
                        end
                    end
                end
            end
        end
        s=size(err);
        Lmin=find(err==min(min(min(min(err)))));
        [i0,j0,k0,m0]=ind2sub(s,Lmin);
        min_err=min(min(min(min(err))));
        Fr1=r1(i0,j0,k0,m0);
        Fr2=r2(i0,j0,k0,m0);
        Fs1=s1(i0,j0,k0,m0);
        Fs2=s2(i0,j0,k0,m0);
        Fdelta_s=delta_s(i0,j0,k0,m0);
        Fr0=r01(j0);
        Fdelta=delta1(i0);
        Flamuta=lamuta(k0);
        FA0=A01(m0);

        filename = "r1=" + Fr1 + ",r2=" + Fr2 + ".csv";
        variable_names = {'Fr1', 'Fr2', 'Fs1', 'Fs2', 'Fr0', 'Fdelta', 'FA0', 'min_err', 'Flamuta'};
        % Fr0>>>qlim   Flamuta>>>qpof   FA0>>>qdp0   Fdelta 需要填入 trans_eq.FA0 文件
        variable_values = [Fr1, Fr2, Fs1, Fs2, Fr0, Fdelta, FA0, min_err, Flamuta];
        fid = fopen(filename, 'w');
        fprintf(fid, '%s,', variable_names{1:end-1});
        fprintf(fid, '%s\n', variable_names{end});
        fprintf(fid, '%.6f,', variable_values(1:end-1));
        fprintf(fid, '%.6f\n', variable_values(end));
        fclose(fid);
    % -----------------------------------------------------------------------------------------------------
    % -----------------------------------------------------------------------------------------------------

        r=0:1/n:1;
        q=qc*(1+(r/Fr0).^(2*Flamuta)).^(1/Flamuta).*(1+FA0*exp(-((r-r_delta)/Fdelta).^2))/(1+FA0);
        dqdr=diff(q)./diff(r);
        dqdr=[dqdr dqdr(end)];
        s=r./q.*dqdr;
        r1=r(min(find(q<2)));
        r2=r(max(find(q<2)));
        dis_r=r2-r1;
        s1=s(min(find(q<2)));
        s2=s(max(find(q<2)));
        plot(r,q)
        hold on
    end
    grid on
    filename = "r2=" + r20 + ".fig";
    saveas(gcf, filename);
    close
end