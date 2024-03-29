%% randomized DLRA Non-linear Schrödinger equation
%% see Projection methods for dynamical low-rank approximation of high-dimensional problems 

    % addpath and cleaning enviroment
    addpath('../rDLR-core')
    clc; clear; rng(123)
K=200;  
N=K
n=K
T=5;

M =diag(1*ones(1,K-1),1) + diag(1*ones(1,K-1),-1);

%% Initial Data:

% Initial value:
sigma=10;
mu1=60;
mu2=50;
nu1=50;
nu2=40;

Y0=zeros(K,K);

for i=1:K
    for j=1:K
        Y0(i,j)=exp(-(i-mu1)^2./(sigma^2)-(j-nu1)^2./(sigma^2))+exp(-(i-mu2)^2./(sigma^2)-(j-nu2)^2./(sigma^2));
    end
end

%% differential equation:
alpha=0.6;
H = @(Y)   1i*(0.5*(M*Y+Y*M)+alpha*(abs(Y).^2).*Y); 


tic
ref= odeSolver(Y0,H,0,T);
toc
F=@(Y,t) H(Y);
stream=RandStream('mt19937ar','Seed',123)
Time=logspace(log10(1e-1), log10(1e-3),8)
err_table_all = []; 

%% randRK with different order (middle plot)
    for funname=["randDLRA_rk_4","randDLRA_rk_3","randDLRA_rk_2"]
            r = 30; %[2,4,8,16,32]
            l = round(0.1*r);  %over-parametrization.
            p = round(0.1*r)
            Omega = randn(N,r+p);
            Psi = randn(N, r+l+p);
            
            X = Y0*Omega; %right-sketch
            Y = Y0'*Psi;  %left-sketch
        
            Y_inital = {X,Y,Omega,Psi};
           % ref = matOdeSolver(matFull(-1,Y0),F,0,T);  
            errTable_randDLRA = [];   
            for dt = Time  
            
                Y_randDLRA = Y_inital;
                maxT = round(T/dt);
                
                for i=1:maxT
                    fun=str2func(funname);
                    
                    Y_randDLRA = fun(Y_randDLRA,F,(i-1)*dt,i*dt,r,stream,"non-constant");
                    %fprintf("r = %d, t = %f \n", r, i*dt);
                end
                ref=odeSolver(Y0,H,0,i*dt);
                err_randDLRA = norm(matFull(1,Y_randDLRA,r) - ref, 'fro');
                errTable_randDLRA = [errTable_randDLRA,err_randDLRA];
                fprintf("randDLRA - dt = %f, err = %e \n", dt, err_randDLRA);
            end

        err_table_all=[err_table_all;errTable_randDLRA];
    end

%% randRK 3 (right plot)
time=logspace(log10(1e-1), log10(1e-4),12)
err_table_all_fixed_rank=[];
rank=[15,20,25,30]
sc = parallel.pool.Constant(RandStream("threefry",Seed=1234)); % set seed
parfor count=1:length(rank)
    r=rank(count)
        stream = sc.Value;        % set each worker seed
        stream.Substream =count;
        p= max(2,round(0.1*r));
        l = max(2,round(0.1*r));  %over-parametrization.
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
        ylim([ymin, ymax]);
        grid on
    set(gca,'FontSize',18)

subplot(1,3,2)
    title('Projected Runge-Kutta')
    loglog(Time, err_table_all(1:3,:).','LineWidth',1.5,'Marker','o')
        hold on
     loglog(Time,(16.*Time).^2,'--','LineWidth',1.5)
      loglog(Time,(7.*Time).^3,'--','LineWidth',1.5)
    loglog(Time,(4.*Time).^4,'--','LineWidth',1.5)
    %yline(norm(ref-U(:,1:r)*sg(1:r,1:r)*V(:,1:r)',"fro"),"LineWidth",1.5);
        
    legend('Location','southeast')
    
    legendStr = [];

    legendStr = [["Rand Rk4","Rand Rk3","Rand Rk2"], "slope 2", "slope 3", "slope 4"];

    legend(legendStr)
    xlabel('\Deltat')
    ylabel('|| Y^{ref} - Y^{approx} ||_F')
    ylim([1e-9 1e0])
    grid on
        set(gca,'FontSize',18)
subplot(1,3,3)
   title('Randomized DLRA')
    loglog(time, err_table_all_fixed_rank(1:length(rank),:).','LineWidth',1.5,'Marker','o')
        hold on
    loglog(time,(5.*time).^3,'--','LineWidth',1.5)
        
    legend('Location','southeast')
    
    legendStr = [];
    for r = rank
        legendStr = [legendStr, "Rand RK3 rank = " + num2str(r)];
    end
    legendStr = [legendStr, "slope 3"];

    legend(legendStr)
    xlabel('\Deltat')
    ylabel('|| Y^{ref} - Y^{approx} ||_F')
    ylim([1e-9 1e0])
    grid on

    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    set(gca,'FontSize',18)
 
