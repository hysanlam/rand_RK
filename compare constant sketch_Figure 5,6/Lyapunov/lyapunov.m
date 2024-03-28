%% randomized DLRA
    % addpath and cleaning enviroment
    addpath('../rDLR-core')
    clc; clear; rng(123);

%% Parameters:
    N = 100;    %Size.
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

    time =logspace(log10(1e-1), log10(1e-3),10);
    trials=50;    
    r = 15; %set rank
    
    sc = parallel.pool.Constant(RandStream("threefry",Seed=1234)); %set seed

    funname_vec=["randDLRA_rk_3","randDLRA_rk_2"]
    err_table_all = zeros(length(funname_vec),length(time),trials); 
parfor count_trials=1:trials
    err_table_all_temp=[];
    stream = sc.Value;        % set each worker seed
    stream.Substream = count_trials;
    for count=1:length(funname_vec)
            
            funname=funname_vec(count);
            l = max(3,round(0.1*r));  %over-parametrization.
            p =  max(3,round(0.1*r));
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
                    Y_randDLRA = fun(Y_randDLRA,F,(i-1)*dt,i*dt,r,stream,"non-constant_sketch"); %need to change for non consant sketch
               
                end               
                ref = integral(@(s) expm((i*dt-s)*A)*C*expm((i*dt-s)*A'),0,i*dt, 'ArrayValued', true,'AbsTol',1e-10)+expm((i*dt)*A)*Y0*expm((i*dt)*A');
                err_randDLRA = norm(matFull(1,Y_randDLRA,r) - ref, 'fro');
                errTable_randDLRA = [errTable_randDLRA,err_randDLRA];
                fprintf("randDLRA - dt = %f, err = %e \n", dt, err_randDLRA);
            end
    
        err_table_all_temp=[err_table_all_temp;errTable_randDLRA];
    end
    err_table_all(:,:,count_trials)=err_table_all_temp;
end

    