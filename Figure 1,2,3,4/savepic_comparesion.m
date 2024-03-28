time=Time
subplot(1,3,1)
  [U,sg,V] = svd(ref);
    ymin = min(diag(sg));
    ymax = max(diag(sg));
    
    title('Singular values reference solution')
    semilogy(diag(sg(1:50,1:50)),'LineWidth',4)
        ylim([ymin, ymax]);
        grid on
    set(gca,'FontSize',16)

subplot(1,3,2)
     title('Projected Runge-Kutta')
    loglog(Time, err_table_all_rand(1:3,:).'./norm(ref,'fro'),'LineWidth',1.5,'Marker','o')
        hold on
    loglog(Time,(70.*Time).^1,'--','LineWidth',1.5)
    hold on
    loglog(Time,(5.*Time).^2,'--','LineWidth',1.5)
    hold on
    loglog(Time,(3.*Time).^3,'--','LineWidth',1.5)

    yline(norm(ref-U(:,1:r)*sg(1:r,1:r)*V(:,1:r)',"fro")./norm(ref,'fro'),"LineWidth",1.5);
        % loglog(time,(5.*time).^3,'--','LineWidth',1)
        % loglog(time,(3.*time).^4,'--','LineWidth',1)
    legend('Location','southeast')
    
    legendStr = [];

    legendStr = [["Randomized RK3","Randomized RK2","Randomized Euler"], "slope 1", "slope 2", "slope 3", "best approximation"];

    legend(legendStr)
    xlabel('\Deltat')
    ylabel('Rel. error Y^{ref} - Y^{approx}')
    ylim([1e-8 ymax])
    grid on

    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    % Get rid of tool bar and pulldown menus that are along top of figure.
    %set(gcf, 'Toolbar', 'none', 'Menu', 'none');
    set(gca,'FontSize',16)
 subplot(1,3,3)
    title('Projected Runge-Kutta')
    loglog(Time, err_table_all(1:3,:).'./norm(ref,'fro'),'LineWidth',1.5,'Marker','o')
        hold on
       loglog(Time,(70.*Time).^1,'--','LineWidth',1.5)
    hold on
    loglog(Time,(5.*Time).^2,'--','LineWidth',1.5)
    hold on
    loglog(Time,(3.*Time).^3,'--','LineWidth',1.5)

    yline(norm(ref-U(:,1:r)*sg(1:r,1:r)*V(:,1:r)',"fro")./norm(ref,'fro'),"LineWidth",1.5);
        % loglog(time,(5.*time).^3,'--','LineWidth',1)
        % loglog(time,(3.*time).^4,'--','LineWidth',1)
    legend('Location','southeast')
    
    legendStr = [];

    legendStr = [["Projected RK3","Projected RK2","Projected Euler"], "slope 1", "slope 2", "slope 3", "best approximation"];

    legend(legendStr)
    xlabel('\Deltat')
    ylabel('Rel. error Y^{ref} - Y^{approx}')
    ylim([1e-8 ymax])
    grid on

    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 0.7]);
    % Get rid of tool bar and pulldown menus that are along top of figure.
    %set(gcf, 'Toolbar', 'none', 'Menu', 'none');
    set(gca,'FontSize',16)
    
    % plot(linspace(0,T,length(ortho_norm)),ortho_norm,"LineWidth",2)
    % xlabel('t')
    % ylabel('$\|(I-P)F\|_F$','Interpreter','latex')
    % set(gca,'FontSize',16)
