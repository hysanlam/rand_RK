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
stream=RandStream('mt19937ar','Seed',123)
Time=logspace(log10(1e-1), log10(1e-3),8)

err_table_all = []; 
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
% err_rk4=[];
%     for dt = [1e-2,1e-3,1e-4,1e-5]
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
figure();

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
    loglog(Time, err_table_all(1:3,:).','LineWidth',1.5,'Marker','o')
        hold on
    loglog(Time,(16.*Time).^2,'--','LineWidth',1.5)
    hold on
    loglog(Time,(8.*Time).^3,'--','LineWidth',1.5)
    hold on
    loglog(Time,(3.*Time).^4,'--','LineWidth',1.5)

    yline(norm(ref-U(:,1:r)*sg(1:r,1:r)*V(:,1:r)',"fro"),"LineWidth",1.5);
        % loglog(time,(5.*time).^3,'--','LineWidth',1)
        % loglog(time,(3.*time).^4,'--','LineWidth',1)
    legend('Location','southeast')
    
    legendStr = [];

    legendStr = [["Projected rk4","Projected rk3","Projected rk2"], "slope 2", "slope 3", "slope 4", "best approximation"];

    legend(legendStr)
    xlabel('\Deltat')
    ylabel('|| Y^{ref} - Y^{approx} ||_F')
    ylim([0.5e-8 ymax])
    grid on

    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    % Get rid of tool bar and pulldown menus that are along top of figure.
    %set(gcf, 'Toolbar', 'auto', 'Menu', 'auto');
    set(gca,'FontSize',18)
    %saveas(gcf,'prk_r16.fig')
    %plot(ortho_norm)
