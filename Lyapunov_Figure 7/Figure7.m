%% randomized DLRA
    % addpath and cleaning enviroment
    addpath('../rDLR-core')
    clc; clear; rng(123);

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
    C = 10*C ./ norm(C,'fro'); %normalized.
    
    F = @(X,t) A*X+X*A'+C;

%% Initial value and reference solution:

    K_x = 1;
    B = normrnd(0,1,[N,1]);
    Y0 = B*B';
    %Y0 = ones(n,n) + 10^-3*randn(n,n);
    Z0 = Y0;

    ref = integral(@(s) expm((T-s)*A)*C*expm((T-s)*A'),0,T, 'ArrayValued', true,'AbsTol',1e-10)+expm((T)*A)*Y0*expm((T)*A');

%% Randomized DLR algorithm with different solvers (middle plot)

    time =logspace(log10(1e-1), log10(1e-4),10);
    
    stream=RandStream('mt19937ar','Seed',123)
    err_table_all = []; 
    for funname=["randDLRA_rk_4","randDLRA_rk_3","randDLRA_rk_2"]
            r = 32; %[2,4,8,16,32]
            l =max(2,round(0.1*r));  %over-parametrization.
            p =max(2,round(0.1*r));
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
                    
                    Y_randDLRA = fun(Y_randDLRA,F,(i-1)*dt,i*dt,r,stream,"non_constant");
                    %fprintf("r = %d, t = %f \n", r, i*dt);
                end
                
                ref = integral(@(s) expm((i*dt-s)*A)*C*expm((i*dt-s)*A'),0,i*dt, 'ArrayValued', true,'AbsTol',1e-10)+expm((i*dt)*A)*Y0*expm((i*dt)*A');
                err_randDLRA = norm(matFull(1,Y_randDLRA,r) - ref, 'fro');
                errTable_randDLRA = [errTable_randDLRA,err_randDLRA];
                fprintf("randDLRA - dt = %f, err = %e \n", dt, err_randDLRA);
            end

        err_table_all=[err_table_all;errTable_randDLRA];
    end

%%Randomized RK3 with different ranks
   time = logspace(log10(1e-1), log10(1e-4),10); %[0.5,1e-1,1e-2,1e-3,1e-4];
    ranks = [4,8,16,32]; %[2,4,8,16,32]

    err_table_all_fixed_rank = []; 
    sc = parallel.pool.Constant(RandStream("threefry",Seed=1234)); % set seed 
    parfor count=1:length(ranks)
            stream = sc.Value;        % set each worker seed
            stream.Substream =count;
            r=ranks(count);
            l = max(2,round(0.1*r));  %over-parametrization.
            p =  max(2,round(0.1*r));
            Omega = randn(stream,N,r+p);
            Psi = randn(stream,N, r+l+p);
            
            X = Y0*Omega; %right-sketch
            Y = Y0'*Psi;  %left-sketch
        
            Y_inital = {X,Y,Omega,Psi};
           % ref = matOdeSolver(matFull(-1,Y0),F,0,T);  
            errTable_randDLRA = [];   
            for dt = time  
            
                Y_randDLRA = Y_inital;
                maxT = round(T/dt);
                for i=1:maxT
                    Y_randDLRA = randDLRA_rk_3(Y_randDLRA,F,(i-1)*dt,i*dt,r,stream,"non constant_sketch");
                    %fprintf("r = %d, t = %f \n", r, i*dt);
                end
                ref = integral(@(s) expm((i*dt-s)*A)*C*expm((i*dt-s)*A'),0,i*dt, 'ArrayValued', true,'AbsTol',1e-10)+expm((i*dt)*A)*Y0*expm((i*dt)*A');
                err_randDLRA = norm(matFull(1,Y_randDLRA,r) - ref, 'fro');
                errTable_randDLRA = [errTable_randDLRA,err_randDLRA];
                fprintf("randDLRA - dt = %f, err = %e \n", dt, err_randDLRA);
            end

       err_table_all_fixed_rank =[err_table_all_fixed_rank;errTable_randDLRA];
    end


%% Plotting
subplot(1,3,1)
    [U,sg,V] = svd(ref);
    ymin = min(diag(sg));
    ymax = max(diag(sg));
    
    title('Singular values reference solution')
    semilogy(diag(sg(1:50,1:50)),'LineWidth',4)
        ylim([ymin, ymax]);
        grid on
    set(gca,'FontSize',18)

subplot(1,3,2)
    title('Projected Runge-Kutta')
    loglog(time, err_table_all(1:3,:).','LineWidth',1,'Marker','o')
        hold on
     loglog(time,(10.*time).^2,'--','LineWidth',1)
      loglog(time,(3.*time).^3,'--','LineWidth',1)
    loglog(time,(2.*time).^4,'--','LineWidth',1)
    yline(norm(ref-U(:,1:r)*sg(1:r,1:r)*V(:,1:r)',"fro"),"LineWidth",1.5);
        
    legend('Location','southeast')
    
    legendStr = [];

    legendStr = [["Rand rk4","Rand rk3","Rand rk2"], "slope 2", "slope 3", "slope 4"];

    legend(legendStr)
    xlabel('\Deltat')
    ylabel('|| Y^{ref} - Y^{approx} ||_F')
    ylim([1e-10 1e1])
    grid on
    set(gca,'FontSize',18)
subplot(1,3,3)
   title('Randomized DLRA')
    loglog(time, err_table_all_fixed_rank(1:length(ranks),:).','LineWidth',1,'Marker','o')
        hold on
    loglog(time,(2.*time).^3,'--','LineWidth',1)
        
    legend('Location','southeast')
    
    legendStr = [];
    for r = ranks
        legendStr = [legendStr, "Rand RK3 rank = " + num2str(r)];
    end
    legendStr = [legendStr, "slope 3"];

    legend(legendStr)
    xlabel('\Deltat')
    ylabel('|| Y^{ref} - Y^{approx} ||_F')
    ylim([ymin ymax])
    grid on

    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    % Get rid of tool bar and pulldown menus that are along top of figure.
    %set(gcf, 'Toolbar', 'none', 'Menu', 'none');
    set(gca,'FontSize',18)
    %saveas(gcf,'randDLRA_diff_solver_r16.fig')
    