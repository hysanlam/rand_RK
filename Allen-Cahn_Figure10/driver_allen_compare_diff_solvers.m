%% allen
%% see Projection methods for dynamical low-rank approximation of high-dimensional problems 

    % addpath and cleaning enviroment
    addpath('../rDLR-core')
    clc; clear; rng(123)
K=256;  
N=K
n=K



T=10;
x=linspace(0, 2*pi, N);
dx=x(2)-x(1);
 a = 1;
    b = -2;
    A = diag(b*ones(1,N)) + diag(a*ones(1,N-1),1) + diag(a*ones(1,N-1),-1); %Discrete Laplacian.
    
M =(1/(dx.^2))*A;


%% Initial Data:

% Initial value:
Y0=zeros(K,K);

for i=1:K
    for j=1:K
        Y0(i,j)=(exp(-(tan(x(i)).^2))+exp(-(tan(x(j)).^2)))*sin(x(i))*sin(x(j))./(1+exp(abs(csc(-x(i)./2)))+exp(abs(csc(-x(j)./2))));
    end
end

eps=0.01;
H = @(Y)   (eps*(M*Y+Y*M)+Y-Y.^3); 


tic
ref= odeSolver(Y0,H,0,T);
toc
F=@(Y,t) H(Y);

Time=[2e-2,1e-2,5e-3,2.5e-3,1e-3];
err_table_all=[];

 for funname=["randDLRA_rk_4","randDLRA_rk_2","randDLRA_euler"]
            r = 35; %[2,4,8,16,32]
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
                    
                    Y_randDLRA = fun(Y_randDLRA,F,(i-1)*dt,i*dt,r);
                    %fprintf("r = %d, t = %f \n", r, i*dt);
                end

                err_randDLRA = norm(matFull(1,Y_randDLRA,r) - ref, 'fro');
                errTable_randDLRA = [errTable_randDLRA,err_randDLRA];
                fprintf("randDLRA - dt = %f, err = %e \n", dt, err_randDLRA);
            end

        err_table_all=[err_table_all;errTable_randDLRA];
    end


% err_rk4=[];
%     for dt = [2e-2,1e-2,5e-3,2.5e-3,1e-3,5e-4]
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
    loglog(Time, err_table_all(1:3,:).','LineWidth',1,'Marker','o')
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

   

