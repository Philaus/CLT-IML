% Calculate magnetic-island widths.

clear
all_files = dir();
dir_flags = [all_files.isdir] & ~strcmp({all_files.name}, '.') & ~strcmp({all_files.name}, '..');
sub_folders = all_files(dir_flags);
plt_case = {sub_folders.name};
plt_case = strcat(plt_case, filesep);

% Initialize the main figure for the aggregate comparison.
figure(1); 
clf;

for i=1:length(plt_case)
    m=2;
    n=1;
    npsi=211;
    nthe=101;
    myt=32;
    nst_start=0;
    nst_end=199;

    n2th=2*(nthe-1);
    fac=1.0/n2th/myt;
    mintor=0;
    maxtor=3;
    minpol=0;
    maxpol=16;
    mtor=maxtor-mintor+1;
    mpol=maxpol-minpol+1;
    dat_name=[char(plt_case(i)),'Brmn'];
    q_p_g=readmatrix([char(plt_case(i)),'q_p_g.dat']);
    psi=q_p_g(:,2);
    q=q_p_g(:,3);
    r=sqrt((psi-min(psi))/(max(psi)-min(psi)));

    % ==== Magnetic shear at the inner and outer q=2 surfaces ====
    q2_idx_in  = find(q < m/n, 1, 'first'); % Inner intersection.
    q2_idx_out = find(q < m/n, 1, 'last');  % Outer intersection.
    dqdr=diff(q)./diff(r);
    dqdr_in  = dqdr(q2_idx_in);
    dqdr_out = dqdr(q2_idx_out);

    time=load([char(plt_case(i)),'nstime.dat']);
    t_end=time(:,4);

    % ==== Initialize the two width arrays ====
    Wt_in_list = zeros(nst_end+1,1);
    Wt_out_list = zeros(nst_end+1,1);
    t_list = zeros(nst_end+1,1);
    
    for nst=nst_start:nst_end
        disp(nst)
        str_nst=num2str(nst,'%.3d');
        wr=load([dat_name,'r',str_nst]);
        wi=load([dat_name,'i',str_nst]);
        Brmnr=zeros(npsi,mpol,mtor);
        Brmni=zeros(npsi,mpol,mtor);
        Brmn=zeros(npsi,mpol,mtor);
        for kn=1:mtor
            Brmnr(:,:,kn)=reshape(wr(:,kn+2),npsi,mpol)*fac;
            Brmni(:,:,kn)=reshape(wi(:,kn+2),npsi,mpol)*fac;
            z = Brmnr(:,:,kn) + 1i * Brmni(:,:,kn);
            Brmn(:,:,kn)=abs(z);
        end
        
        Brmn_in = Brmn(q2_idx_in, m+1, n+1);
        Brmn_out = Brmn(q2_idx_out, m+1, n+1);

        Wt_in  = 4 * sqrt(m * Brmn_in / (n * n * abs(dqdr_in))) * 2 * sqrt(2*pi);
        Wt_out = 4 * sqrt(m * Brmn_out / (n * n * abs(dqdr_out))) * 2 * sqrt(2*pi);
        
        Wt_in_list(nst+1) = Wt_in;
        Wt_out_list(nst+1) = Wt_out;
        t_list(nst+1) = t_end(nst+1);
    end
    
    % Create the table and write it to the current folder.
    data_table = table(t_list, Wt_in_list, Wt_out_list, 'VariableNames', {'Time', 'Wt_Inner', 'Wt_Outer'});
    csv_filename = [char(plt_case(i)), sub_folders(i).name, '_Wt_data.csv'];
    writetable(data_table, csv_filename);
    
    % ==== Plot and save the figure for the current case ====
    fig_case = figure('Visible', 'off'); % Keep the new figure hidden.
    plot(t_list, Wt_in_list, 'b-', 'LineWidth', 2, 'DisplayName', 'Inner $q=2$');
    hold on;
    plot(t_list, Wt_out_list, 'r-', 'LineWidth', 2, 'DisplayName', 'Outer $q=2$');
    xlabel('$t$', 'Interpreter', 'latex', 'FontSize', 18);
    ylabel('$W_T$', 'Interpreter', 'latex', 'FontSize', 18);
    title(['Case: ', sub_folders(i).name], 'Interpreter', 'none');
    legend('Interpreter', 'latex', 'Location', 'best', 'FontSize', 14);
    grid on;
    
    % Save the image in the corresponding folder.
    img_filename = [char(plt_case(i)), sub_folders(i).name, '_Wt_plot.png'];
    exportgraphics(fig_case, img_filename, 'Resolution', 300);
    close(fig_case); % Close the temporary figure.
    
    figure(1); % Return focus to the aggregate figure.
    leg = sub_folders(i).name;
    % Add this case's usually dominant outer-island width to the aggregate plot.
    plot(t_list, Wt_out_list, 'LineWidth', 2, 'DisplayName', [leg, ' (Outer)'])
    hold on
    
    xlabel('$t$','Interpreter','latex','FontSize',18)
    xlim([min(t_list),max(t_list)])
    ylabel('$W_T$','Interpreter','latex','FontSize',18)
    legend('Interpreter','none','Location','best','FontSize',12)
    grid on
end
