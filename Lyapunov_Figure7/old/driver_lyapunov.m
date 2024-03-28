%% randomized DLRA
    % addpath and cleaning enviroment
    addpath('../rDLR-core')
    clc; clear; rng(123)

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

%% Randomized DLR algorithm

    time = logspace(log10(1e-1), log10(1e-4),12); %[0.5,1e-1,1e-2,1e-3,1e-4];
    ranks = [4,8,16,32]; %[2,4,8,16,32]

    err_table_all = []; 
    sc = parallel.pool.Constant(RandStream("threefry",Seed=1234)); % set seed 
    parfor count=1:length(ranks)
            stream = sc.Value;        % set each worker seed
            stream.Substream =count;
            r=ranks(count);
            l = round(0.1*r);  %over-parametrization.
            p = round(0.1*r)
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

        err_table_all=[err_table_all;errTable_randDLRA];
    end

%% Plotting

subplot(1,2,1)
    sg = svd(ref);
    ymin = min(sg);
    ymax = max(sg);
    
    title('Singular values reference solution')
    semilogy(sg(1:50),'LineWidth',4)
        ylim([ymin, ymax]);
        grid on
    set(gca,'FontSize',18)

subplot(1,2,2)
    title('Randomized DLRA')
    loglog(time, err_table_all(1:length(ranks),:).','LineWidth',1,'Marker','o')
        hold on
    loglog(time,(2.*time).^3,'--','LineWidth',1)
        
    legend('Location','southeast')
    
    legendStr = [];
    for r = ranks
        legendStr = [legendStr, "rDLR rank = " + num2str(r)];
    end
    legendStr = [legendStr, "slope 3"];

    legend(legendStr)
    xlabel('\Deltat')
    ylabel('|| Y^{ref} - Y^{approx} ||_F')
    ylim([ymin ymax])
    grid on

    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    % Get rid of tool bar and pulldown menus that are along top of figure.
    set(gcf, 'Toolbar', 'none', 'Menu', 'none');
    set(gca,'FontSize',18)

    %saveas(gcf,'randDLRA.fig')
