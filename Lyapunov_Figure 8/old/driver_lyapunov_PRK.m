%% randomized DLRA
    % addpath and cleaning enviroment
    addpath('../rDLR-core')
    clc; clear; rng(123)

%% Parameters:
    N = 500;    %Size.
    T = .5;     %Final time.

    a = 1;
    b = -2;
    A = diag(b*ones(1,N)) + diag(a*ones(1,N-1),1) + diag(a*ones(1,N-1),-1); %Discrete Laplacian.
    
    K = 10;
    B = normrnd(0,1,[N,K]);
    B=orth(B);
    sig = diag(10.^-(0:K-1));
    C = B*sig*B';
    C = 0.01*C ./ norm(C,'fro'); %normalized.
    
    F = @(X,t) A*X+X*A'+C;

%% Initial value and reference solution:

    K_x = 1;
    B = normrnd(0,1,[N,1]);
    Y0 = B*B';
    %Y0 = ones(n,n) + 10^-3*randn(n,n);
    Z0 = Y0;

    ref = integral(@(s) expm((T-s)*A)*C*expm((T-s)*A'),0,T, 'ArrayValued', true,'AbsTol',1e-10)+expm((T)*A)*Y0*expm((T)*A');

%% Randomized DLR algorithm

    time = [1e-1,5e-2,2.5e-2,1e-2,5e-3,2.5e-3,1e-3]; %[0.5,1e-1,1e-2,1e-3,1e-4];
    ranks = [4,8,16,32,40,50,60]; %[2,4,8,16,32]

    err_table_all = []; 
    parfor count=1:length(ranks)
            r=ranks(count);
            [U,S,V]=svd(Y0);
         
             
            Y_inital = {U(:,1:r),S(1:r,1:r),V(:,1:r)};
           % ref = matOdeSolver(matFull(-1,Y0),F,0,T);  
            errTable_randDLRA = [];   
            for dt = time  
            
                Y_projected = Y_inital;
                maxT = round(T/dt);
                for i=1:maxT
                    Y_projected = projected_rk3(Y_projected,F,(i-1)*dt,i*dt,r);
                    %fprintf("r = %d, t = %f \n", r, i*dt);
                end

                err_randDLRA = norm(Y_projected{1}*Y_projected{2}*Y_projected{3}' - ref, 'fro');
                errTable_randDLRA = [errTable_randDLRA,err_randDLRA];
                fprintf("randDLRA - dt = %f, err = %e \n", dt, err_randDLRA);
            end

        err_table_all=[err_table_all;errTable_randDLRA];
    end

%% Plotting

subplot(1,2,1)
    sg = svd(ref);
    ymin = min(sg);
    ymax = max(sg);
    
    title('Singular values reference solution')
    semilogy(sg(1:50),'LineWidth',4)
        ylim([ymin, ymax]);
        grid on
    set(gca,'FontSize',18)

subplot(1,2,2)
    title('Randomized DLRA')
    loglog(time, err_table_all(1:length(ranks),:).','LineWidth',1,'Marker','o')
        hold on
    loglog(time,(5.*time).^4,'--','LineWidth',1)
     hold on
    loglog(time,(10*time).^2,'--','LineWidth',1)
        
    legend('Location','southeast')
    
    legendStr = [];
    for r = ranks
        legendStr = [legendStr, "PRK4 rank = " + num2str(r)];
    end
    legendStr = [legendStr, "slope 4","slope 2"];

    legend(legendStr)
    xlabel('\Deltat')
    ylabel('|| Y^{ref} - Y^{approx} ||_F')
    ylim([ymin ymax])
    grid on

    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    % Get rid of tool bar and pulldown menus that are along top of figure.
    set(gcf, 'Toolbar', 'none', 'Menu', 'none');
    set(gca,'FontSize',18)

    saveas(gcf,'randDLRA_prk4.eps')
