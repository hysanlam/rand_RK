%% randomized DLRA
    % addpath and cleaning enviroment
    addpath('../rDLR-core')
    clc; clear; rng(123)

%% Parameters:
     N = 500;    %Size.
    T = ;     %Final time.

    a = 1;
    b = -2;
    A = diag(b*ones(1,N)) + diag(a*ones(1,N-1),1) + diag(a*ones(1,N-1),-1); %Discrete Laplacian.
    K = 10;
    B = normrnd(0,1,[N,K]);
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

    time = [5e-2]; %[0.5,1e-1,1e-2,1e-3,1e-4];
    r= 40; %[2,4,8,16,32]

    err_table_all = []; 
   
            for dt = time  
            
               
                maxT = round(T/dt);
                for i=1:maxT
                    t=i*dt
                   sol=integral(@(s) expm((t-s)*A)*C*expm((t-s)*A'),0,t, 'ArrayValued', true,'AbsTol',1e-10)+expm((t)*A)*Y0*expm((t)*A');
                    [U,S,V]=svd(sol);
                    x0 = {U(:,1:r),S(1:r,1:r),V(:,1:r)};  
                    temp=norm(F(U*S*V',t)-calculate_pf(x0,F,t),"fro")
                    err_table_all =[ err_table_all ,temp];
                    i
                end

     
            end

%% Plotting

plot(err_table_all )
