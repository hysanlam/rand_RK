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

alpha=0.6;
H = @(Y)   1i*(0.5*(M*Y+Y*M)+alpha*(abs(Y).^2).*Y); 


tic
ref= odeSolver(Y0,H,0,T);
toc
F=@(Y,t) H(Y);

Time=logspace(log10(1e-1), log10(1e-4),12)
err_table_all=[];
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
    for dt = Time
        Y_randDLRA = Y_inital;        
        for i=1:(T/dt)
            Y_randDLRA = randDLRA_rk_3(Y_randDLRA,F,(i-1)*dt,i*dt,r,stream,"non constant_sketch");
        end
        ref=odeSolver(Y0,H,0,i*dt);
        err_randDLRA = norm(matFull(1,Y_randDLRA,r) - ref, 'fro');
        errTable_randDLRA = [errTable_randDLRA,err_randDLRA];
        fprintf("randDLRA - dt = %f, err = %e \n", dt, err_randDLRA);
    end
    err_table_all=[err_table_all;errTable_randDLRA]
end
% err_rk4=[];
%     for dt = [1e-1,1e-2,1e-3,1e-4]
% 
%         Z=Y0;
% 
%         for i=1:(T/dt)
%             Z=rk4(Z, F, dt);
% 
%         end
%         err_rk4=[err_rk4,norm(Z - ref, 'fro')]
%     end


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
    loglog(Time,1*(Time).^4,'--','LineWidth',1)
        
    legend('Location','southeast')
    
    legendStr = [];
    for r = rank
        legendStr = [legendStr, "rDLR rank = " + num2str(r)];
    end
    legendStr = [legendStr, "slope 4"];

    legend(legendStr)
    xlabel('\Deltat')
    ylabel('|| Y^{ref} - Y^{approx} ||_F')
    ylim([1e-13 1])
    xlim([min(Time),max(Time)])
    grid on

    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    % Get rid of tool bar and pulldown menus that are along top of figure.
    set(gcf, 'Toolbar', 'none', 'Menu', 'none');
    set(gca,'FontSize',18)

    %saveas(gcf,'randDLRA_rk4.fig')
