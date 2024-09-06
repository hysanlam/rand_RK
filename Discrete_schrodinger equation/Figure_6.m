
% addpath and cleaning enviroment
addpath('../randRK-core')
clc; clear; rng(123)
K=2^9;
N=K
n=K
numer_trials=10;

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
ref= odeSolver(Z0,H,0,T)
toc
F=@(Y,t) H(Y);

time=logspace(log10(2.5e-1), log10(5e-4),14);
time=T./round(T./time);
sc = parallel.pool.Constant(RandStream("threefry",Seed=123)); % set seed
err_table_all=[];
fun_name_rand=["rand_rk_4","rand_rk_3","rand_rk_2","rand_euler"];
err_table_all = [];
err_table_all_mutiple=zeros(length(fun_name_rand),length(time),numer_trials);
for trial=1:numer_trials
    err_table_all = [];
    parfor funname=1:length(fun_name_rand)
        stream = sc.Value;        % set each worker seed
        stream.Substream =funname+(trial)*length(fun_name_rand);
        r = 40; %[2,4,8,16,32]
        l = max(2,round(0.1*r));  %over-parametrization.
        p =  max(2,round(0.1*r));
        Omega = randn(stream,N,r+p);
        Psi = randn(stream,N, r+l+p);

        X = Y0*Omega; %right-sketch
        Y = Y0'*Psi;  %left-sketch

        Y_inital = {X,Y,Omega,Psi};
        % ref = matOdeSolver(matFull(-1,Y0),F,0,T);
        errTable_randRK = [];
        for dt = time

            Y_randRK = Y_inital;
            maxT = round(T/dt);

            for i=1:maxT
                fun=str2func(fun_name_rand(funname));

                Y_randRK = fun(Y_randRK,F,(i-1)*dt,i*dt,r,stream,"non_constant");
                %fprintf("r = %d, t = %f \n", r, i*dt);
            end
            ref= odeSolver(Z0,H,0,i*dt);
            err_randRK = norm(matFull(1,Y_randRK,r) - ref, 'fro');
            errTable_randRK = [errTable_randRK,err_randRK];
            fprintf("randRK - dt = %f, err = %e \n", dt, err_randRK);
        end

        err_table_all=[err_table_all;errTable_randRK];
    end
    err_table_all_mutiple(:,:,trial)=err_table_all;
end



err_table_all_fixed_rank=[];
rank=[20,30,40,50];
sc = parallel.pool.Constant(RandStream("threefry",Seed=1234)); % set seed
err_table_all_fixed_rank_mutiple=zeros(length(rank),length(time),numer_trials);
for trial=1:numer_trials
    err_table_all_fixed_rank=[];
    parfor count=1:length(rank)
        r=rank(count)
        stream = sc.Value;        % set each worker seed
        stream.Substream =count+(trial)*length(rank);
        p=max(2,round(0.1*r))
        l = max(2,round(0.1*r));  %over-parametrization.
        Omega = randn(stream,n,r+p);
        Psi = randn(stream,n, r+l+p);

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
            ref=odeSolver(Y0,H,0,i*dt);
            err_randRK = norm(matFull(1,Y_randRK,r) - ref, 'fro');
            errTable_randRK = [errTable_randRK,err_randRK];
            fprintf("randRK - dt = %f, err = %e \n", dt, err_randRK);
        end
        err_table_all_fixed_rank=[err_table_all_fixed_rank;errTable_randRK]
    end
    err_table_all_fixed_rank_mutiple(:,:,trial)=err_table_all_fixed_rank;
end
%% Plotting
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
m=mean(err_table_all_mutiple,3).';
[min_vec,max_vec] = bounds(err_table_all_mutiple,3);
errorbar(time, m, min_vec.'-m,max_vec.'-m,'LineWidth',2,MarkerSize=8)
set(gca, 'XScale','log', 'YScale','log')
hold on
loglog(time,(0.04.*time).^1,'--','LineWidth',1)
loglog(time,(0.1.*time).^2,'--','LineWidth',1)

loglog(time,(0.25.*time).^4,'--','LineWidth',1)
%yline(norm(ref-U(:,1:r)*sg(1:r,1:r)*V(:,1:r)',"fro"),"LineWidth",1.5);

legend('Location','southeast')

legendStr = [];

legendStr = [["Rand Rk4","Rand Rk3","Rand Rk2","Rand Euler"], "slope 1","slope 2", "slope 4"];

legend(legendStr)
xlabel('\Deltat')
ylabel('|| Y^{ref} - Y^{approx} ||_F')
ylim([1e-10 5e-2])
xlim([min(time),max(time)])
grid on
set(gca,'FontSize',18)
subplot(1,3,3)
m=mean(err_table_all_fixed_rank_mutiple,3).';
[min_vec,max_vec] = bounds(err_table_all_fixed_rank_mutiple,3);
errorbar(time, m, min_vec.'-m,max_vec.'-m,'LineWidth',2,MarkerSize=8)
set(gca, 'XScale','log', 'YScale','log')
hold on
hold on
loglog(time,(0.25.*time).^4,'--','LineWidth',1)

legend('Location','southeast')

legendStr = [];
for r = rank(1:length(rank))
    legendStr = [legendStr, "Rand RK4 rank = " + num2str(r)];
end
legendStr = [legendStr, "slope 4"];

legend(legendStr)
xlabel('\Deltat')
ylabel('|| Y^{ref} - Y^{approx} ||_F')
ylim([1e-10 5e-2])
xlim([min(time),max(time)])
grid on

set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
%set(gcf, 'Toolbar', 'none', 'Menu', 'none');
set(gca,'FontSize',18)
%saveas(gcf,'randDLRA_diff_solver_r16.fig')
