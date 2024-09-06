%% Compare with PRK


% addpath and cleaning enviroment
addpath('../randRK-core')
clc; clear; rng(123);
numer_trials=10;
%% Parameters:

N = 128;    %Size.
x_1=linspace(-pi,pi,N);
x_2=linspace(-pi,pi,N);
[X,Y] = meshgrid(x_1,x_2);
dx=x_1(2)-x_1(1);

% change case here
%% case 1:
alpha=[1]%1%1e-1%1e-5;
%% case 2:
%alpha=.05;

a = 1;
b = -2;
A = diag(b*ones(1,N)) + diag(a*ones(1,N-1),1) + diag(a*ones(1,N-1),-1); %Discrete Laplacian.
C=zeros(N,N);
for k=1:11
    C=C+10^(-k+1).*exp(-k*(X.^2+Y.^2));
end
C=C./norm(C,"fro");
%F = @(X,t)(A*X+X*A')+alpha*C;

%% Initial value and reference solution:
T = 1;    %old: .5 %Final time.
% Y0=sin(x_1)'*sin(x_2);
% [U,S,V]=svd(Y0);
% S(2:13,2:13)=diag(5*10.^([-7:-.5:-12.5]));
% Y0=U(:,1:13)*S(1:13,1:13)*V(:,1:13)';
% Z0 =Y0;

S=zeros(20,20)
S(1:20,1:20)=(pi./dx)*diag([1,5*10.^( ([-7:-.5:-16]))]);
U=zeros(N,20);
for i=1:20
    U(:,i)=sqrt(dx./pi)*sin(i*x_1);
end
Y0=U*S*U';
%Y0=odeSolver(Y0,F,0,0.2);


time =logspace(log10(2e-1), log10(.5e-3),10);
time=T./round(T./time);

err_table_all = [];

%% RRK

ortho_norm=[];
ref_best_error=[];
sc = parallel.pool.Constant(RandStream("threefry",Seed=1234)); % set seed


F = @(X,t)(A*X+X*A')+alpha*C;
ref = odeSolver(Y0,F,0,T);
rank_rk4 = [5,10,15,20]%[5,10,14,18,20]; %[2,4,8,16,32]
[U_ref,sg_ref,V_ref] = svd(ref);
%ref_best_error=[ref_best_error,norm(ref-U_ref(:,1:r)*sg_ref(1:r,1:r)*V_ref(:,1:r)',"fro")];
err_table_all_rk4_mutiple=zeros(length(rank_rk4),length(time),numer_trials)
for trial=1:numer_trials
    err_table_all_rk4=zeros(length(rank_rk4),length(time));

    parfor count=1:length(rank_rk4)
        stream = sc.Value;        % set each worker seed
        stream.Substream =count+trial*length(rank_rk4);

        r=rank_rk4(count)
        p=max(2,round(0.1*r));
        l = max(2,round(0.1*r)); %over-parametrization.
        Omega = randn(stream,N,r+p);
        Psi = randn(stream,N, r+l+p);

        X = Y0*Omega; %right-sketch
        Y = Y0'*Psi;  %left-sketch

        Y_inital = {X,Y,Omega,Psi};
        % ref = matOdeSolver(matFull(-1,Y0),F,0,T);
        errTable_randRK = [];
        for dt = time
            Y_randRK = Y_inital;
            for i=1:(T/dt)
                Y_randRK = rand_rk_4(Y_randRK,F,(i-1)*dt,i*dt,r,stream,"non constant_sketch");
            end
            %ref = odeSolver(Y0,F,0,i*dt);
            err_randRK = norm(matFull(1,Y_randRK,r) - ref, 'fro');
            errTable_randRK = [errTable_randRK,err_randRK];
            fprintf("randRK - dt = %f, err = %e \n", dt, err_randRK);
        end
        err_table_all_rk4(count,:)=errTable_randRK;
    end
    err_table_all_rk4_mutiple(:,:,trial)=err_table_all_rk4;
end

sc = parallel.pool.Constant(RandStream("threefry",Seed=12345)); % set seed
rank_euler = [2,3,5,8]; %[2,4,8,16,32]
err_table_all_euler_mutiple=zeros(length(rank_euler),length(time),numer_trials)
for trial=1:numer_trials
err_table_all_euler=zeros(length(rank_euler),length(time));


parfor count=1:length(rank_euler)
    stream = sc.Value;        % set each worker seed
    stream.Substream =count+trial*length(rank_euler);

    r=rank_euler(count)
    p=max(2,round(0.1*r));
    l = max(2,round(0.1*r)); %over-parametrization.
    Omega = randn(stream,N,r+p);
    Psi = randn(stream,N, r+l+p);

    X = Y0*Omega; %right-sketch
    Y = Y0'*Psi;  %left-sketch

    Y_inital = {X,Y,Omega,Psi};
    % ref = matOdeSolver(matFull(-1,Y0),F,0,T);
    errTable_randRK = [];
    for dt = time
        Y_randRK = Y_inital;
        for i=1:(T/dt)
            Y_randRK = rand_euler(Y_randRK,F,(i-1)*dt,i*dt,r,stream,"non constant_sketch");
        end
        ref = odeSolver(Y0,F,0,i*dt);
        err_randRK = norm(matFull(1,Y_randRK,r) - ref, 'fro');
        errTable_randRK = [errTable_randRK,err_randRK];
        fprintf("randRK - dt = %f, err = %e \n", dt, err_randRK);
    end
    err_table_all_euler(count,:)=errTable_randRK;
end
err_table_all_euler_mutiple(:,:,trial)=err_table_all_euler;
end

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
m=mean(err_table_all_euler_mutiple,3).';
[min_vec,max_vec] = bounds(err_table_all_euler_mutiple,3)
errorbar(time, m, min_vec.'-m,max_vec.'-m,'LineWidth',1)
hold on
set(gca, 'XScale','log', 'YScale','log')
loglog(time,(0.05.*time).^1,'--','LineWidth',1)
legend('Location','southeast')
legend('Location','southeast')
legendStr = [];
for r = rank_euler(1:length(rank_euler))
    legendStr = [legendStr, "Rand Euler rank = " + num2str(r)];
end
legendStr = [legendStr, "slope 1"];
legend(legendStr)
xlabel('\Deltat')
ylabel('|| Y^{ref} - Y^{approx} ||_F')
ylim([1e-5 5e-2])
xlim([min(time),max(time)])
grid on
set(gca,'FontSize',18)

subplot(1,3,3)
title('Randomized RK')
m=mean(err_table_all_rk4_mutiple,3).';
[min_vec,max_vec] = bounds(err_table_all_rk4_mutiple,3);
errorbar(time, m, min_vec.'-m,max_vec.'-m,'LineWidth',1)
set(gca, 'XScale','log', 'YScale','log')
hold on
loglog(time,(0.25.*time).^4,'--','LineWidth',1)
legend('Location','southeast')
legendStr = [];
for r = rank_rk4(1:length(rank_rk4))
    legendStr = [legendStr, "Rand RK4 rank = " + num2str(r)];
end
legendStr = [legendStr, "slope 4"];

legend(legendStr)
xlabel('\Deltat')
ylabel('|| Y^{ref} - Y^{approx} ||_F')
ylim([1e-11 1e-3])
xlim([min(time),max(time)])
grid on

set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
%set(gcf, 'Toolbar', 'none', 'Menu', 'none');
set(gca,'FontSize',18)