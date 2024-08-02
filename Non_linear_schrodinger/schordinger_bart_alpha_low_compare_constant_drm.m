%% Compare with PRK


% addpath and cleaning enviroment
addpath('../rDLR-core')
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


%% Randomized DLR algorithm

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
        errTable_randDLRA = [];
        for dt = time
            Y_randDLRA = Y_inital;
            for i=1:(T/dt)
                Y_randDLRA = randDLRA_rk_4(Y_randDLRA,F,(i-1)*dt,i*dt,r,stream,"non_constant_complex");
            end

            err_randDLRA = norm(matFull(1,Y_randDLRA,r) - ref, 'fro');
            errTable_randDLRA = [errTable_randDLRA,err_randDLRA];
            fprintf("randDLRA - dt = %f, err = %e \n", dt, err_randDLRA);
        end
        err_table_all_rk4(count,:)=errTable_randDLRA;
    end
    err_table_all_rk4_mutiple(:,:,trial)=err_table_all_rk4;
end

sc = parallel.pool.Constant(RandStream("threefry",Seed=12345)); % set seed
err_table_all_rk4_mutiple_constant=zeros(length(rank),length(time),numer_trials)
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
        errTable_randDLRA = [];
        for dt = time
            Y_randDLRA = Y_inital;
            for i=1:(T/dt)
                Y_randDLRA = randDLRA_rk_4(Y_randDLRA,F,(i-1)*dt,i*dt,r,stream,"constant_sketch_complex");
            end

            err_randDLRA = norm(matFull(1,Y_randDLRA,r) - ref, 'fro');
            errTable_randDLRA = [errTable_randDLRA,err_randDLRA];
            fprintf("randDLRA - dt = %f, err = %e \n", dt, err_randDLRA);
        end
        err_table_all_rk4(count,:)=errTable_randDLRA;
    end
    err_table_all_rk4_mutiple_constant(:,:,trial)=err_table_all_rk4;
end


rank_rk2 = [15,18,21,24]; %[2,4,8,16,32]
sc = parallel.pool.Constant(RandStream("threefry",Seed=123456)); % set seed
err_table_all_rk2_mutiple=zeros(length(rank),length(time),numer_trials)
for trial=1:numer_trials
    err_table_all_rk2=zeros(length(rank),length(time));

    parfor count=1:length(rank)
        stream = sc.Value;        % set each worker seed
        stream.Substream =count+trial*length(rank);

        r=rank_rk2(count)
        p=max(2,round(0.1*r));
        l = max(2,round(0.1*r)); %over-parametrization.
        Omega = randn(stream,N,r+p);
        Psi = randn(stream,N, r+l+p);

        X = Y0*Omega; %right-sketch
        Y = Y0'*Psi;  %left-sketch

        Y_inital = {X,Y,Omega,Psi};
        % ref = matOdeSolver(matFull(-1,Y0),F,0,T);
        errTable_randDLRA = [];
        for dt = time
            Y_randDLRA = Y_inital;
            for i=1:(T/dt)
                Y_randDLRA = randDLRA_rk_2(Y_randDLRA,F,(i-1)*dt,i*dt,r,stream,"non_constant_complex");
            end

            err_randDLRA = norm(matFull(1,Y_randDLRA,r) - ref, 'fro');
            errTable_randDLRA = [errTable_randDLRA,err_randDLRA];
            fprintf("randDLRA - dt = %f, err = %e \n", dt, err_randDLRA);
        end
        err_table_all_rk2(count,:)=errTable_randDLRA;
    end
    err_table_all_rk2_mutiple(:,:,trial)=err_table_all_rk2;
end

sc = parallel.pool.Constant(RandStream("threefry",Seed=1234567)); % set seed
err_table_all_rk2_mutiple_constant=zeros(length(rank),length(time),numer_trials)
for trial=1:numer_trials
    err_table_all_rk2=zeros(length(rank),length(time));

    parfor count=1:length(rank)
        stream = sc.Value;        % set each worker seed
        stream.Substream =count+trial*length(rank);

        r=rank_rk2(count)
        p=max(2,round(0.1*r));
        l = max(2,round(0.1*r)); %over-parametrization.
        Omega = randn(stream,N,r+p);
        Psi = randn(stream,N, r+l+p);

        X = Y0*Omega; %right-sketch
        Y = Y0'*Psi;  %left-sketch

        Y_inital = {X,Y,Omega,Psi};
        % ref = matOdeSolver(matFull(-1,Y0),F,0,T);
        errTable_randDLRA = [];
        for dt = time
            Y_randDLRA = Y_inital;
            for i=1:(T/dt)
                Y_randDLRA = randDLRA_rk_2(Y_randDLRA,F,(i-1)*dt,i*dt,r,stream,"constant_sketch_complex");
            end

            err_randDLRA = norm(matFull(1,Y_randDLRA,r) - ref, 'fro');
            errTable_randDLRA = [errTable_randDLRA,err_randDLRA];
            fprintf("randDLRA - dt = %f, err = %e \n", dt, err_randDLRA);
        end
        err_table_all_rk2(count,:)=errTable_randDLRA;
    end
    err_table_all_rk2_mutiple_constant(:,:,trial)=err_table_all_rk2;
end

subplot(1,4,1)

m=mean(err_table_all_rk2_mutiple,3).';
[min_vec,max_vec] = bounds(err_table_all_rk2_mutiple,3)
errorbar(time, m, min_vec.'-m,max_vec.'-m,'LineWidth',2)
hold on
set(gca, 'XScale','log', 'YScale','log')
loglog(time,(0.05.*time).^2,'--','LineWidth',1)
legend('Location','southeast')
legend('Location','southeast')
legendStr = [];
for r = rank_rk2(1:length(rank_rk2))
    legendStr = [legendStr, "Rank = " + num2str(r)];
end
legendStr = [legendStr, "slope 2"];
legend(legendStr)
xlabel('h')
ylabel('|| Y^{ref} - Y ||_F')
grid on
title('Rand RK2')
set(gca,'FontSize',14)

subplot(1,4,2)

m=mean(err_table_all_rk2_mutiple_constant,3).';
[min_vec,max_vec] = bounds(err_table_all_rk2_mutiple_constant,3)
errorbar(time, m, min_vec.'-m,max_vec.'-m,'LineWidth',2)
hold on
set(gca, 'XScale','log', 'YScale','log')
loglog(time,(0.05.*time).^2,'--','LineWidth',1)
legend('Location','southeast')
legend('Location','southeast')
legendStr = [];
for r = rank_rk2(1:length(rank_rk2))
    legendStr = [legendStr, "Rank = " + num2str(r)];
end
legendStr = [legendStr, "slope 2"];
legend(legendStr)
xlabel('h')
ylabel('|| Y^{ref} - Y ||_F')
grid on
title('Rand RK2 with constant random matrix')
set(gca,'FontSize',14)

subplot(1,4,3)

m=mean(err_table_all_rk4_mutiple,3).';
[min_vec,max_vec] = bounds(err_table_all_rk4_mutiple,3)
errorbar(time, m, min_vec.'-m,max_vec.'-m,'LineWidth',2)
hold on
set(gca, 'XScale','log', 'YScale','log')
loglog(time,(0.05.*time).^1,'--','LineWidth',1)
legend('Location','southeast')
legend('Location','southeast')
legendStr = [];
for r = rank(1:length(rank))
    legendStr = [legendStr, "Rank = " + num2str(r)];
end
legendStr = [legendStr, "slope 4"];
legend(legendStr)
xlabel('h')
ylabel('|| Y^{ref} - Y ||_F')
ylim([1e-11 1e-3])
xlim([min(time),max(time)])
grid on
title('Rand RK4')
set(gca,'FontSize',14)

subplot(1,4,4)

m=mean(err_table_all_rk4_mutiple_constant,3).';
[min_vec,max_vec] = bounds(err_table_all_rk4_mutiple_constant,3);
errorbar(time, m, min_vec.'-m,max_vec.'-m,'LineWidth',2)
set(gca, 'XScale','log', 'YScale','log')
hold on
loglog(time,(0.25.*time).^4,'--','LineWidth',1)
legend('Location','southeast')
legendStr = [];
for r = rank(1:length(rank))
    legendStr = [legendStr, "Rank= " + num2str(r)];
end
legendStr = [legendStr, "slope 4"];

legend(legendStr)
xlabel('h')
ylabel('|| Y^{ref} - Y||_F')
ylim([1e-11 1e-3])
xlim([min(time),max(time)])
grid on
title('Rand RK4 with constant random matrix')
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
%set(gcf, 'Toolbar', 'none', 'Menu', 'none');
set(gca,'FontSize',14)