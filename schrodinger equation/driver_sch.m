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
ref= odeSolver(Z0,H,0,T);
toc
F=@(Y,t) H(Y);

%logspace(log10(5e-1), log10(1e-4),10)
Time=logspace(log10(5e-1), log10(1e-4),10);

err_table_all=[];
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
    err_table_all=[err_table_all;errTable_randDLRA]
end


 err_rk4=[];

    % for dt = Time
    % 
    %     Z=Z0;
    % 
    %     for i=1:(T/dt)
    %         Z=rk4(Z, F, dt,(i-1)*dt,i*dt);
    % 
    %     end
    %     err_rk4=[err_rk4,norm(Z - ref, 'fro')];
    % end


%% Plotting

subplot(1,2,1)
    sg = svd(ref);
    ymin = min(sg);
    ymax = max(sg);
    
    title('Singular values reference solution')
    semilogy(sg(1:70),'LineWidth',4)
        ylim([1e-14, ymax]);
        grid on
    set(gca,'FontSize',18)

subplot(1,2,2)
    title('Randomized DLRA')
    loglog(Time, err_table_all(1:length(rank),:).','LineWidth',1,'Marker','o')
        hold on
    loglog(Time,(0.3.*Time).^4,'--','LineWidth',1)
        
    legend('Location','southeast')
    
    legendStr = [];
    for r = rank
        legendStr = [legendStr, "rDLR rank = " + num2str(r)];
    end
    legendStr = [legendStr, "slope 4"];

    legend(legendStr)
    xlabel('\Deltat')
    ylabel('|| Y^{ref} - Y^{approx} ||_F')
    ylim([1e-12 1e-2])
    xlim([min(Time),max(Time)])
    grid on

    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    % Get rid of tool bar and pulldown menus that are along top of figure.
    set(gcf, 'Toolbar', 'none', 'Menu', 'none');
    set(gca,'FontSize',18)

    %saveas(gcf,'randDLRA_rk4.fig')
