%% randomized DLRA Non-linear Schrödinger equation- compare with PRK
%% see Projection methods for dynamical low-rank approximation of high-dimensional problems for the example
%% We compare two cases: 
%% case 1: alpha=0.1, r=8, Time=logspace(log10(5e-1), log10(7.5e-4),12)
%% case 2: alpha=0.6, r=25, Time= logspace(log10(5e-2), log10(7.5e-4),10)


% addpath and cleaning enviroment
addpath('../../rDLR-core')
clc; clear; rng(123)

% parameter:
K=100;  
N=K
n=K
T=5;

M =diag(1*ones(1,K-1),1) + diag(1*ones(1,K-1),-1);

%%Change cases here:
%% case 1:
 alpha=0.1; r=8; Time=logspace(log10(5e-1), log10(7.5e-4),12);
%% case 2: 
% alpha=0.6; r=25; Time= logspace(log10(5e-2), log10(7.5e-4),10);

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



H = @(Y)   1i*(0.5*(M*Y+Y*M)+alpha*(abs(Y).^2).*Y); 


tic
ref= odeSolver(Y0,H,0,T);
toc
F=@(Y,t) H(Y);
stream=RandStream('mt19937ar','Seed',123)


err_table_all = []; 
    for funname=["randDLRA_rk_3","randDLRA_rk_2","randDLRA_euler"]    
            l = max(2,round(0.1*r));  %over-parametrization.
            p = max(2,round(0.1*r));
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

 %% PRK
 err_table_all_prk=[];
 ortho_norm=[];
 for funname=["projected_rk3","projected_rk2","projected_rk1"]
             
            [U,S,V]=svd(Y0);
            Y_inital = {U(:,1:r),S(1:r,1:r),V(:,1:r)};
         
           % ref = matOdeSolver(matFull(-1,Y0),F,0,T);  
            errTable_prk = [];   
            for dt = Time  
            
               Y_projected= Y_inital;
                maxT = round(T/dt);
                
                for i=1:maxT
                    fun=str2func(funname);
                     
                    Y_projected = fun(Y_projected,F,(i-1)*dt,i*dt,r);
                    if funname=="projected_rk3" & dt== 1e-3
                        temp=norm(F(Y_projected{1}*Y_projected{2}*Y_projected{3}',i*dt)-calculate_pf(Y_projected,F,i*dt),"fro");
                        ortho_norm=[ortho_norm,temp];
                    end
                    %fprintf("r = %d, t = %f \n", r, i*dt);
                end
                ref = odeSolver(Y0,H,0,i*dt);

                err_prk = norm(Y_projected{1}*Y_projected{2}*Y_projected{3}' - ref, 'fro');
                errTable_prk = [errTable_prk,err_prk];
                fprintf("randDLRA - dt = %f, err = %e \n", dt, err_prk);
            end

         err_table_all_prk=[ err_table_all_prk;errTable_prk];
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
    loglog(Time, err_table_all(1:3,:).','LineWidth',1,'Marker','o')
        hold on
    loglog(Time,(27.*Time).^1,'--','LineWidth',1)
    loglog(Time,(7.*Time).^2,'--','LineWidth',1)
    loglog(Time,(2.*Time).^3,'--','LineWidth',1)
    yline(norm(ref-U(:,1:r)*sg(1:r,1:r)*V(:,1:r)',"fro"),"LineWidth",1.5);
        
    legend('Location','southeast')
    
    legendStr = [];

    legendStr = [["Rand RK3","Rand RK2","Rand Euler"], "slope 1", "slope 2", "slope 3","Best approximation"];

    legend(legendStr)
    xlabel('\Deltat')
    ylabel('|| Y^{ref} - Y^{approx} ||_F')
    ylim([5e-5 5e1])
    grid on

    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    % Get rid of tool bar and pulldown menus that are along top of figure.
    set(gcf, 'Toolbar', 'none', 'Menu', 'none');
    set(gca,'FontSize',18)

subplot(1,3,3)
    title('Projected Runge-Kutta')
    loglog(Time, err_table_all_prk(1:3,:).','LineWidth',1,'Marker','o')
        hold on
    loglog(Time,(27.*Time).^1,'--','LineWidth',1)
    loglog(Time,(7.*Time).^2,'--','LineWidth',1)
    loglog(Time,(2.*Time).^3,'--','LineWidth',1)
    yline(norm(ref-U(:,1:r)*sg(1:r,1:r)*V(:,1:r)',"fro"),"LineWidth",1.5);
        
    legend('Location','southeast')
    
    legendStr = [];

    legendStr = [["Projected RK3","Projected RK2","Projected Euler"], "slope 1", "slope 2", "slope 3","Best approximation"];

    legend(legendStr)
    xlabel('\Deltat')
    ylabel('|| Y^{ref} - Y^{approx} ||_F')
    ylim([5e-5 5e1])
    grid on

    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    % Get rid of tool bar and pulldown menus that are along top of figure.
    set(gcf, 'Toolbar', 'none', 'Menu', 'none');
    set(gca,'FontSize',18)