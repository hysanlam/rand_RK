%% Compare with PRK


% addpath and cleaning enviroment
addpath('../randRK-core')
clc; clear; rng(123);
numer_trials=10;
%% Parameters:
K=100;
N=K
n=K
T=5;
M =diag(1*ones(1,K-1),1) + diag(1*ones(1,K-1),-1);

% change case here
%% case 1:
alpha=[3e-1]%1%1e-1%1e-5;
%% case 2:
%alpha=.05;



%% Initial value and reference solution:
T = 5;    %old: .5 %Final time.
sigma=10;
mu1=60;
mu2=50;
nu1=50;
nu2=40;

Y0_inital=zeros(K,K);

for i=1:K
    for j=1:K
        Y0(i,j)=exp(-(i-mu1)^2./(sigma^2)-(j-nu1)^2./(sigma^2))+exp(-(i-mu2)^2./(sigma^2)-(j-nu2)^2./(sigma^2));
    end
end
[U,S,V]=svd(Y0);
U_perp=orth(eye(K,K)-U(:,1:2)*U(:,1:2)')
V_perp=orth(eye(K,K)-V(:,1:2)*V(:,1:2)')

S(3:32,3:32)=diag(10.^(-9*ones(32-2,1)));
Y0=[U(:,1:2),U_perp(:,1:30)]*S(1:32,1:32)*[V(:,1:2),V_perp(:,1:30)]'
%Y0=odeSolver(Y0,F,0,0.2);

time =logspace(log10(5e-2), log10(2.5e-4),9);
%time =.5e-3
time=T./round(T./time);

err_table_all = [];

%% RRK

ortho_norm=[];
ref_best_error=[];
sc = parallel.pool.Constant(RandStream("threefry",Seed=1234)); % set seed


H = @(Y)   1i*(0.5*(M*Y+Y*M)+alpha*(abs(Y).^2).*Y);
F=@(Y,t) H(Y);
ref = odeSolver(Y0,F,0,T);
rank = [15,20,25,30]; %[2,4,8,16,32]
[U_ref,sg_ref,V_ref] = svd(ref);
%ref_best_error=[ref_best_error,norm(ref-U_ref(:,1:r)*sg_ref(1:r,1:r)*V_ref(:,1:r)',"fro")];
err_table_all_rk4_mutiple=zeros(length(rank),length(time),numer_trials)
for trial=1:numer_trials
    err_table_all_rk4=zeros(length(rank),length(time));

    parfor count=1:length(rank)
        stream = sc.Value;        % set each worker seed
        stream.Substream =count+trial*length(rank);

        r=rank(count)
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
                Y_randRK = rand_rk_4(Y_randRK,F,(i-1)*dt,i*dt,r,stream,"non_constant_complex");
            end

            err_randRK = norm(matFull(1,Y_randRK,r) - ref, 'fro');
            errTable_randRK = [errTable_randRK,err_randRK];
            fprintf("randRK - dt = %f, err = %e \n", dt, err_randRK);
        end
        err_table_all_rk4(count,:)=errTable_randRK;
    end
    err_table_all_rk4_mutiple(:,:,trial)=err_table_all_rk4;
end

sc = parallel.pool.Constant(RandStream("threefry",Seed=12345)); % set seed
rank = [2,4,6,8,12]; %[2,4,8,16,32]

err_table_all_euler_mutiple=zeros(length(rank),length(time),numer_trials)
for trial=1:numer_trials
    err_table_all_euler=zeros(length(rank),length(time));
    parfor count=1:length(rank)
        stream = sc.Value;        % set each worker seed
        stream.Substream =count+trial*length(rank);

        r=rank(count)
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
                Y_randRK = rand_euler(Y_randRK,F,(i-1)*dt,i*dt,r,stream,"non_constant_complex");
            end

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
loglog(time,(0.1.*time).^1,'--','LineWidth',1)

%yline(norm(ref-U(:,1:r)*sg(1:r,1:r)*V(:,1:r)',"fro"),"LineWidth",1.5);

legend('Location','southeast')

legendStr = [];
for r = [2,4,6,8,12]
    legendStr = [legendStr, "Rand Euler rank = " + num2str(r)];
end
legend(legendStr)
xlabel('\Deltat')
ylabel('|| Y^{ref} - Y^{approx} ||_F')
ylim([1e-2 20])
loglog(time,(100*time).^1,'--','LineWidth',1)
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
loglog(time,(1.5*time).^4,'--','LineWidth',1)

legend('Location','southeast')

legendStr = [];
for r = [15,20,25,30]
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