%% Compare with PRK 
%% here we compare two cases, alpha=0.01  and alpha=10 (need to change manually below):
    

    % addpath and cleaning enviroment
    addpath('../rDLR-core')
    clc; clear; rng(123)

%% Parameters:
    N = 100;    %Size.
    T = .5;     %Final time.

    %%Change cases here:
    %% case 1:
    alpha=0.01;
    %% case 2: 
    % alpha=10;


    a = 1;
    b = -2;
    A = diag(b*ones(1,N)) + diag(a*ones(1,N-1),1) + diag(a*ones(1,N-1),-1); %Discrete Laplacian.
    K = 10;
    B = normrnd(0,1,[N,K]);
    B=orth(B);
    sig = diag(10.^-(0:K-1));
    C = B*sig*B';
    C = alpha*C ./ norm(C,'fro'); %normalized. or 0.01
    
    F = @(X,t) A*X+X*A'+C;

%% Initial value and reference solution:

    K_x = 1;
    B = normrnd(0,1,[N,1]);
    Y0 = B*B';
    %Y0 = ones(n,n) + 10^-3*randn(n,n);
    Z0 = Y0;

    ref = integral(@(s) expm((T-s)*A)*C*expm((T-s)*A'),0,T, 'ArrayValued', true,'AbsTol',1e-10)+expm((T)*A)*Y0*expm((T)*A');

%% Randomized DLR algorithm

    time =logspace(log10(1e-1), log10(1e-3),10); %[0.5,1e-1,1e-2,1e-3,1e-4];
            
    ortho_norm=[];
    err_table_all = []; 
    for funname=["projected_rk3","projected_rk2","projected_rk1"]
            r = 15; %[2,4,8,16,32]
             
            [U,S,V]=svd(Y0);
            Y_inital = {U(:,1:r),S(1:r,1:r),V(:,1:r)};
         
           % ref = matOdeSolver(matFull(-1,Y0),F,0,T);  
            errTable_randDLRA = [];   
            for dt = time  
            
               Y_projected= Y_inital;
                maxT = round(T/dt);
                
                for i=1:maxT
                    fun=str2func(funname);
                     
                    Y_projected = fun(Y_projected,F,(i-1)*dt,i*dt,r);
                    if funname=="projected_rk3" & dt== 1e-3
                        temp=norm(F(Y_projected{1}*Y_projected{2}*Y_projected{3}',i*dt)-calculate_pf(Y_projected,F,i*dt),"fro")
                        ortho_norm=[ortho_norm,temp];
                    end
                    %fprintf("r = %d, t = %f \n", r, i*dt);
                end
                ref = integral(@(s) expm((i*dt-s)*A)*C*expm((i*dt-s)*A'),0,i*dt, 'ArrayValued', true,'AbsTol',1e-10)+expm((i*dt)*A)*Y0*expm((i*dt)*A');

                err_randDLRA = norm(Y_projected{1}*Y_projected{2}*Y_projected{3}' - ref, 'fro');
                errTable_randDLRA = [errTable_randDLRA,err_randDLRA];
                fprintf("randDLRA - dt = %f, err = %e \n", dt, err_randDLRA);
            end

        err_table_all=[err_table_all;errTable_randDLRA];
    end

%% Plotting
figure;
subplot(1,2,1)
    [U,sg,V] = svd(ref);
    ymin = min(diag(sg));
    ymax = max(diag(sg));
    
    title('Singular values reference solution')
    semilogy(diag(sg(1:50,1:50)),'LineWidth',4)
        ylim([ymin, ymax]);
        grid on
    set(gca,'FontSize',18)

subplot(1,2,2)
    title('Projected Runge-Kutta')
    loglog(time, err_table_all(1:3,:).','LineWidth',1.5,'Marker','o')
        hold on
    loglog(time,(10.*time).^2,'--','LineWidth',1)
    yline(norm(ref-U(:,1:r)*sg(1:r,1:r)*V(:,1:r)',"fro"),"LineWidth",1.5);
        loglog(time,(5.*time).^3,'--','LineWidth',1)
        % loglog(time,(3.*time).^4,'--','LineWidth',1)
    legend('Location','southeast')
    
    legendStr = [];

    legendStr = [["Projected Rk3","Projected Rk2","Projected Euler"], "slope 2", "best approximation"];

    legend(legendStr)
    xlabel('\Deltat')
    ylabel('|| Y^{ref} - Y^{approx} ||_F')
    ylim([1e-9 ymax])
    grid on

    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 0.6 0.6]);
    % Get rid of tool bar and pulldown menus that are along top of figure.
    %set(gcf, 'Toolbar', 'none', 'Menu', 'none');
    set(gca,'FontSize',18)
    %saveas(gcf,'prk_r16.fig')
    %plot(ortho_norm)
