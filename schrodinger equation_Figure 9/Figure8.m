%% randomized DLRA
    % addpath and cleaning enviroment
    addpath('../rDLR-core')
    clc; clear; rng(123)
K=2^9;  
N=K
n=K


%dt=1e-2;
T=0.5;

%% Build Potential Matrices:
global V_cos

D = -K/2 : K/2-1;
dx = (2*pi*K^-1); 
x = dx.*D;

V_cos = diag(1-cos(x));
M = diag(2*ones(1,K)) + diag(-1*ones(1,K-1),1) + diag(-1*ones(1,K-1),-1);

%% Initial Data:

% Initial value:
    U0 = orth(rand(K,K));
    S0 = diag(10.^(flip(-K:-1))); 

    V0 = orth(rand(K,K));
    Y0=U0*S0*V0';


H = @(Y)   0.5*(M*Y+Y*M)-V_cos*Y*V_cos.'; 

Z0=Y0;
tic
ref= odeSolver(Z0,H,0,T)
toc
F=@(Y,t) H(Y);

time=logspace(log10(2.5e-1), log10(5e-4),14);
stream=RandStream('mt19937ar','Seed',123)
err_table_all=[];

 err_table_all = []; 
    for funname=["randDLRA_rk_4","randDLRA_rk_3","randDLRA_rk_2"]
            r = 40; %[2,4,8,16,32]
            l = round(0.1*r);  %over-parametrization.
            p = round(0.1*r)
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
                ref= odeSolver(Z0,H,0,i*dt);
                err_randDLRA = norm(matFull(1,Y_randDLRA,r) - ref, 'fro');
                errTable_randDLRA = [errTable_randDLRA,err_randDLRA];
                fprintf("randDLRA - dt = %f, err = %e \n", dt, err_randDLRA);
            end

        err_table_all=[err_table_all;errTable_randDLRA];
    end

Time=logspace(log10(5e-1), log10(1e-4),10);

err_table_all_fixed_rank=[];
rank=[10,20,30,40,50];
sc = parallel.pool.Constant(RandStream("threefry",Seed=1234)); % set seed 
parfor count=1:length(rank)
    stream = sc.Value;        % set each worker seed
    stream.Substream =count;
    
    r=rank(count)
    p=round(0.2*r)
    l = round(0.2*r);  %over-parametrization.
    Omega = randn(stream,n,r+p);
    Psi = randn(stream,n, r+l+p);
    
    X = Y0*Omega; %right-sketch
    Y = Y0'*Psi;  %left-sketch
    
    Y_inital = {X,Y,Omega,Psi};
       % ref = matOdeSolver(matFull(-1,Y0),F,0,T);  
    errTable_randDLRA = [];   
    for dt = Time
        Y_randDLRA = Y_inital;        
        for i=1:(T/dt)
            Y_randDLRA = randDLRA_rk_3(Y_randDLRA,F,(i-1)*dt,i*dt,r,stream,"non constant_sketch");
        end
        ref= odeSolver(Z0,H,0,i*dt);
        err_randDLRA = norm(matFull(1,Y_randDLRA,r) - ref, 'fro');
        errTable_randDLRA = [errTable_randDLRA,err_randDLRA];
        fprintf("randDLRA - dt = %f, err = %e \n", dt, err_randDLRA);
    end
    err_table_all_fixed_rank=[err_table_all_fixed_rank;errTable_randDLRA]
end


time=logspace(log10(1e-1), log10(1e-4),12)
err_table_all_fixed_rank=[];
rank=[15,20,25,30]
sc = parallel.pool.Constant(RandStream("threefry",Seed=1234)); % set seed
parfor count=1:length(rank)
    r=rank(count)
        stream = sc.Value;        % set each worker seed
        stream.Substream =count;
        p=round(0.1*r)
        l = round(0.1*r);  %over-parametrization.
        Omega = randn(stream,n,r+p);
        Psi = randn(stream,n, r+l+p);

        X = Y0*Omega; %right-sketch
        Y = Y0'*Psi;  %left-sketch

        Y_inital = {X,Y,Omega,Psi};
       % ref = matOdeSolver(matFull(-1,Y0),F,0,T);  
     errTable_randDLRA = [];   
    for dt = time
        Y_randDLRA = Y_inital;        
        for i=1:(T/dt)
            Y_randDLRA = randDLRA_rk_3(Y_randDLRA,F,(i-1)*dt,i*dt,r,stream,"non constant_sketch");
        end
        ref=odeSolver(Y0,H,0,i*dt);
        err_randDLRA = norm(matFull(1,Y_randDLRA,r) - ref, 'fro');
        errTable_randDLRA = [errTable_randDLRA,err_randDLRA];
        fprintf("randDLRA - dt = %f, err = %e \n", dt, err_randDLRA);
    end
    err_table_all_fixed_rank=[err_table_all_fixed_rank;errTable_randDLRA]
end
%% Plotting
subplot(1,3,1)
    [U,sg,V] = svd(ref);
    ymin = min(diag(sg));
    ymax = max(diag(sg));
    
    title('Singular values reference solution')
    semilogy(diag(sg(1:50,1:50)),'LineWidth',4)
        ylim([1e-12, ymax]);
        grid on
    set(gca,'FontSize',18)

subplot(1,3,2)
    title('Projected Runge-Kutta')
    loglog(time, err_table_all(1:3,:).','LineWidth',1,'Marker','o')
        hold on
     loglog(time,(0.1.*time).^2,'--','LineWidth',1)
      loglog(time,(0.25.*time).^3,'--','LineWidth',1)
    loglog(time,(0.25.*time).^4,'--','LineWidth',1)
    %yline(norm(ref-U(:,1:r)*sg(1:r,1:r)*V(:,1:r)',"fro"),"LineWidth",1.5);
        
    legend('Location','southeast')
    
    legendStr = [];

    legendStr = [["Rand Rk4","Rand Rk3","Rand Rk2"], "slope 2", "slope 3", "slope 4"];

    legend(legendStr)
    xlabel('\Deltat')
    ylabel('|| Y^{ref} - Y^{approx} ||_F')
    ylim([1e-10 5e-2])
    xlim([min(time),max(time)])
    grid on
        set(gca,'FontSize',18)
subplot(1,3,3)
   title('Randomized DLRA')
    loglog(Time, err_table_all_fixed_rank(2:length(rank),:).','LineWidth',1,'Marker','o')
        hold on
    loglog(Time,(0.25.*Time).^3,'--','LineWidth',1)
        
    legend('Location','southeast')
    
    legendStr = [];
    for r = rank(2:length(rank))
        legendStr = [legendStr, "Rand RK3 rank = " + num2str(r)];
    end
    legendStr = [legendStr, "slope 3"];

    legend(legendStr)
    xlabel('\Deltat')
    ylabel('|| Y^{ref} - Y^{approx} ||_F')
    ylim([1e-10 5e-2])
    xlim([min(Time),max(Time)])
    grid on

    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    % Get rid of tool bar and pulldown menus that are along top of figure.
    %set(gcf, 'Toolbar', 'none', 'Menu', 'none');
    set(gca,'FontSize',18)
    %saveas(gcf,'randDLRA_diff_solver_r16.fig')
