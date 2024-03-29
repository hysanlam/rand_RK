%% randomized DLRA
    % addpath and cleaning enviroment
    addpath('../rDLR-core')
    clc; clear; rng(123);

%% Parameters:
    N = 100;    %Size.
    T = .5;     %Final time.

    a = 1;
    b = -2;
    A = diag(b*ones(1,N)) + diag(a*ones(1,N-1),1) + diag(a*ones(1,N-1),-1); %Discrete Laplacian.
    K = 10;
    B = normrnd(0,1,[N,K]);
    B=orth(B);
    sig = diag(10.^-(0:K-1));
    C = B*sig*B';
    C = 10*C ./ norm(C,'fro'); %normalized.
    
    F = @(X,t) A*X+X*A'+C;

%% Initial value and reference solution:

    K_x = 1;
    B = normrnd(0,1,[N,1]);
    Y0 = B*B';
    %Y0 = ones(n,n) + 10^-3*randn(n,n);
    Z0 = Y0;

    ref = odeSolver(Y0,F,0,T);
%% Randomized DLR algorithm using non-constant drm

    time =logspace(log10(1e-1), log10(1e-3),10);
    trials=50;    
    r = 15; %set rank
    
    sc = parallel.pool.Constant(RandStream("threefry",Seed=1234)); %set seed

    funname_vec=["randDLRA_rk_3","randDLRA_rk_2"]
    err_table_all = zeros(length(funname_vec),length(time),trials); 
    parfor count_trials=1:trials
        err_table_all_temp=[];
        stream = sc.Value;        % set each worker seed
        stream.Substream = count_trials;
        for count=1:length(funname_vec)
                
                funname=funname_vec(count);
                l = 2;  %over-parametrization.
                p =  2;
                Omega = randn(N,r+p);
                Psi = randn(N, r+l+p);
                
                X = Y0*Omega; %right-sketch
                Y = Y0'*Psi;  %left-sketch
            
                Y_inital = {X,Y,Omega,Psi};
               % ref = matOdeSolver(matFull(-1,Y0),F,0,T);  
                errTable_randDLRA = [];   
                for dt = time             
                    Y_randDLRA = Y_inital;
                    maxT = round(T/dt);                
                    for i=1:maxT           
    
                        fun=str2func(funname);
                        Y_randDLRA = fun(Y_randDLRA,F,(i-1)*dt,i*dt,r,stream,"non-constant_sketch"); %need to change for non consant sketch
                   
                    end               
                    ref = odeSolver(Y0,F,0,i*dt);
                    err_randDLRA = norm(matFull(1,Y_randDLRA,r) - ref, 'fro');
                    errTable_randDLRA = [errTable_randDLRA,err_randDLRA];
                    fprintf("randDLRA - dt = %f, err = %e \n", dt, err_randDLRA);
                end
        
            err_table_all_temp=[err_table_all_temp;errTable_randDLRA];
        end
        err_table_all(:,:,count_trials)=err_table_all_temp;
    end
    
    %% Randomized DLR algorithm using constant drm
    
    err_table_all_constant = zeros(length(funname_vec),length(time),trials); 
    parfor count_trials=1:trials
        err_table_all_temp=[];
        stream = sc.Value;        % set each worker seed
        stream.Substream = count_trials;
        for count=1:length(funname_vec)
                
                funname=funname_vec(count);
                l = 2;  %over-parametrization.
                p =  2;
                Omega = randn(N,r+p);
                Psi = randn(N, r+l+p);
                
                X = Y0*Omega; %right-sketch
                Y = Y0'*Psi;  %left-sketch
            
                Y_inital = {X,Y,Omega,Psi};
               % ref = matOdeSolver(matFull(-1,Y0),F,0,T);  
                errTable_randDLRA = [];   
                for dt = time             
                    Y_randDLRA = Y_inital;
                    maxT = round(T/dt);                
                    for i=1:maxT           
    
                        fun=str2func(funname);
                        Y_randDLRA = fun(Y_randDLRA,F,(i-1)*dt,i*dt,r,stream,"constant_sketch"); 
                   
                    end               
                    ref = odeSolver(Y0,F,0,i*dt);
                    err_randDLRA = norm(matFull(1,Y_randDLRA,r) - ref, 'fro');
                    errTable_randDLRA = [errTable_randDLRA,err_randDLRA];
                    fprintf("randDLRA - dt = %f, err = %e \n", dt, err_randDLRA);
                end
        
            err_table_all_temp=[err_table_all_temp;errTable_randDLRA];
        end
        err_table_all_constant(:,:,count_trials)=err_table_all_temp;
    end



%% plotting
Time=time
[U,sg,V] = svd(ref);
 ymin = min(diag(sg));
    ymax = max(diag(sg));
    
subplot(1,2,1)
     title('Projected Runge-Kutta')
     m=mean(err_table_all_constant(1,:,:),3).';
     p95=prctile(err_table_all_constant(1,:,:),95,3).'-m;
     p5=m-prctile(err_table_all_constant(1,:,:),5,3).';
     errorbar(time, m,p5,p95,'LineWidth',1)
        hold on
     m=mean(err_table_all_constant(2,:,:),3).';
     p95=prctile(err_table_all_constant(2,:,:),95,3).'-m;
     p5=m-prctile(err_table_all_constant(2,:,:),5,3).';
     errorbar(time, m,p5,p95,'LineWidth',1)
     
     hold on
   % errorbar(time, mean(err_table_all(3,:,:),3).',std(err_table_all(3,:,:),[],3).','LineWidth',1)
   %      hold on

     set(gca, 'XScale','log', 'YScale','log')
    % loglog(Time,(70.*Time).^1,'--','LineWidth',1.5)
    % hold on
    % loglog(Time,(5.*Time).^2,'--','LineWidth',1.5)
    % hold on
    % loglog(Time,(3.*Time).^3,'--','LineWidth',1.5)

    yline(norm(ref-U(:,1:r)*sg(1:r,1:r)*V(:,1:r)',"fro"),"LineWidth",1.5);
        % loglog(time,(5.*time).^3,'--','LineWidth',1)
        % loglog(time,(3.*time).^4,'--','LineWidth',1)
    legend('Location','southeast')
    
    legendStr = [];

    legendStr = [["Randomized RK3 Constant","Randomized RK2 Constant"], "best approximation"];

    legend(legendStr)
    xlabel('\Deltat')
    ylabel('Abs. error Y^{ref} - Y^{approx}')
    %ylabel('Rel. error Y^{ref} - Y^{approx}')
    ylim([1e-6 5])
    grid on

    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 0.7 0.7]);
    % Get rid of tool bar and pulldown menus that are along top of figure.
    %set(gcf, 'Toolbar', 'none', 'Menu', 'none');
    set(gca,'FontSize',16)
 subplot(1,2,2)
       title('Projected Runge-Kutta')
    m=mean(err_table_all(1,:,:),3).';
     p95=prctile(err_table_all(1,:,:),95,3).'-m;
     p5=m-prctile(err_table_all(1,:,:),5,3).';
     errorbar(time, m,p5,p95,'LineWidth',1)
        hold on
     m=mean(err_table_all(2,:,:),3).';
     p95=prctile(err_table_all(2,:,:),95,3).'-m;
     p5=m-prctile(err_table_all(2,:,:),5,3).';
     errorbar(time, m,p5,p95,'LineWidth',1)
   % errorbar(time, mean(err_table_all(3,:,:),3).',std(err_table_all(3,:,:),[],3).','LineWidth',1)
   %      hold on

     set(gca, 'XScale','log', 'YScale','log')
    % loglog(Time,(70.*Time).^1,'--','LineWidth',1.5)
    % hold on
    % loglog(Time,(5.*Time).^2,'--','LineWidth',1.5)
    % hold on
    % loglog(Time,(3.*Time).^3,'--','LineWidth',1.5)

    yline(norm(ref-U(:,1:r)*sg(1:r,1:r)*V(:,1:r)',"fro"),"LineWidth",1.5);
        % loglog(time,(5.*time).^3,'--','LineWidth',1)
        % loglog(time,(3.*time).^4,'--','LineWidth',1)
    legend('Location','southeast')
    
    legendStr = [];

    legendStr = [["Randomized RK3","Randomized RK2"], "best approximation"];

    legend(legendStr)
    xlabel('\Deltat')
    ylabel('Abs. error Y^{ref} - Y^{approx}')
    %ylabel('Rel. error Y^{ref} - Y^{approx}')
    ylim([1e-6 5])
    grid on

    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 0.7 0.7]);
    % Get rid of tool bar and pulldown menus that are along top of figure.
    %set(gcf, 'Toolbar', 'none', 'Menu', 'none');
    set(gca,'FontSize',16)

    