%% Compare with PRK


% addpath and cleaning enviroment
addpath('../rDLR-core')
clc; clear; rng(123);

%% Parameters:
numer_trials=10;
N = 128;    %Size.
x_1=linspace(-pi,pi,N);
x_2=linspace(-pi,pi,N);
[X,Y] = meshgrid(x_1,x_2);
dx=x_1(2)-x_1(1);

% change case here
%% case 1:
alpha=[1e-5,1]%1%1e-1%1e-5;
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

%% Randomized DLR algorithm

time =logspace(log10(2e-1), log10(.5e-3),10);
%time =.5e-3
time=T./round(T./time);

err_table_all = [];

%% RRK
test_method_rrk=["randDLRA_rk_4","randDLRA_rk_2","randDLRA_euler"]

ortho_norm=[];
ref_best_error=[];
sc = parallel.pool.Constant(RandStream("threefry",Seed=1234)); % set seed
err_table_all_rrk_mutiple=zeros(length(test_method_rrk),length(time),length(alpha),numer_trials);
for trial=1:numer_trials
    for alpha_count=1:length(alpha)

        err_table_all_rrk_temp=[];

        F = @(X,t)(A*X+X*A')+alpha(alpha_count)*C;
        ref = odeSolver(Y0,F,0,T);
        r = 10; %[2,4,8,16,32]
        [U_ref,sg_ref,V_ref] = svd(ref);
        ref_best_error=[ref_best_error,norm(ref-U_ref(:,1:r)*sg_ref(1:r,1:r)*V_ref(:,1:r)',"fro")];

        l = max(2,round(0.1*r));  %over-parametrization.
        p = max(2,round(0.1*r));

        parfor  funname=1:length(test_method_rrk)
            stream = sc.Value;        % set each worker seed
            stream.Substream =length(test_method_rrk)*(alpha_count-1)+(trial-1)*length(test_method_rrk)*length(alpha)+funname;

            Omega = randn(stream,N,r+p);
            Psi = randn(stream,N, r+l+p);

            X = Y0*Omega; %right-sketch
            Y = Y0'*Psi;  %left-sketch

            Y_inital = {X,Y,Omega,Psi};

            % ref = matOdeSolver(matFull(-1,Y0),F,0,T);
            errTable_rrk = [];
            for dt = time

                Y_randDLRA= Y_inital;
                maxT = round(T/dt);

                for i=1:maxT
                    fun=str2func(test_method_rrk(funname));

                    Y_randDLRA = fun(Y_randDLRA,F,(i-1)*dt,i*dt,r,stream,"non_constant");
                    % if funname=="projected_rk3" & dt== .5e-3
                    %     temp=norm(F(Y_projected{1}*Y_projected{2}*Y_projected{3}',i*dt)-calculate_pf(Y_projected,F,i*dt),"fro");
                    %     ortho_norm=[ortho_norm,temp];
                    % end
                    %fprintf("r = %d, t = %f \n", r, i*dt);
                end
                ref = odeSolver(Y0,F,0,i*dt); err_rrk = norm(matFull(1,Y_randDLRA,r) - ref, 'fro');
                errTable_rrk = [errTable_rrk,err_rrk];
                fprintf("RRK - dt = %f, err = %e \n", dt, err_rrk);
            end

            err_table_all_rrk_temp=[ err_table_all_rrk_temp;errTable_rrk];
        end
        err_table_all_rrk_mutiple(:,:,alpha_count,trial)= err_table_all_rrk_temp;
    end
end

test_method_prk=["projected_rk4","projected_rk2","projected_rk1"]
err_table_all_prk=zeros(length(test_method_prk),length(time),length(alpha));
ortho_norm_all=[];
for alpha_count=1:length(alpha)
    err_table_all_prk_temp=[]
    
    F = @(X,t)(A*X+X*A')+alpha(alpha_count)*C;
    ref = odeSolver(Y0,F,0,T);
    r = 10;

    parfor  funname=1:length(test_method_prk)

        [U,S,V]=svd(Y0);
        Y_inital = {U(:,1:r),S(1:r,1:r),V(:,1:r)};
        errTable_prk = [];
         ortho_norm=[];
        for dt = time

            Y_projected= Y_inital;
            maxT = round(T/dt);

            for i=1:maxT
                fun=str2func(test_method_prk(funname));

                Y_projected = fun(Y_projected,F,(i-1)*dt,i*dt,r);
                if test_method_prk((funname))=="projected_rk2" & dt== .5e-3
                    temp=norm(F(Y_projected{1}*Y_projected{2}*Y_projected{3}',i*dt)-calculate_pf(Y_projected,F,i*dt),"fro");
                    ortho_norm=[ortho_norm,temp];
                end
                
                fprintf("r = %d, t = %f \n", r, i*dt);
            end
            ortho_norm_all=[ortho_norm_all;ortho_norm];
            ref = odeSolver(Y0,F,0,i*dt); err_prk = norm(Y_projected{1}*Y_projected{2}*Y_projected{3}' - ref, 'fro');
            errTable_prk = [errTable_prk,err_prk];
            fprintf("PRK - dt = %f, err = %e \n", dt, err_prk);
        end

        err_table_all_prk_temp=[ err_table_all_prk_temp;errTable_prk];
    end
    err_table_all_prk(:,:,alpha_count)= err_table_all_prk_temp
end

%%%%%%%%%% Projector splitting %%%%%%%%%%%
test_method=["matProjSplit"]
err_table_all_proj=zeros(length(test_method),length(time),length(alpha));
ortho_norm=[];
ref_best_error=[];
for alpha_count=1:length(alpha)
    err_table_all_proj_temp=[];
    F = @(X,t)(A*X+X*A')+alpha(alpha_count)*C;
    ref = odeSolver(Y0,F,0,T);
    [U_ref,sg_ref,V_ref] = svd(ref);
    ref_best_error=[ref_best_error,norm(ref-U_ref(:,1:r)*sg_ref(1:r,1:r)*V_ref(:,1:r)',"fro")];

    r = 10; %[2,4,8,16,32]
    parfor funname=1:length(test_method)


        [U,S,V]=svd(Y0);
        Y_inital = {U(:,1:r),V(:,1:r),S(1:r,1:r)};

        % ref = matOdeSolver(matFull(-1,Y0),F,0,T);
        errTable_proj = [];
        for dt = time

            Y_projected= Y_inital;
            maxT = round(T/dt);

            for i=1:maxT
                fun=str2func(test_method(funname));

                Y_projected = fun(Y_projected,F,(i-1)*dt,i*dt);

                %fprintf("r = %d, t = %f \n", r, i*dt);
            end
            ref = odeSolver(Y0,F,0,i*dt);
            err_proj = norm(Y_projected{1}*Y_projected{3}*Y_projected{2}' - ref, 'fro');
            errTable_proj = [errTable_proj,err_proj];
            fprintf("projectorsplitting - dt = %f, err = %e \n", dt, err_proj);
        end

        err_table_all_proj_temp=[ err_table_all_proj_temp;errTable_proj];
    end
    err_table_all_proj(:,:,alpha_count)=err_table_all_proj_temp
end
%% Plotting

subplot(1,4,1)

%loglog(time, err_table_all_rrk(:,:,1).','LineWidth',1,'Marker','o')
m=mean(err_table_all_rrk_mutiple(:,:,1,:),4).';
[min_vec,max_vec] = bounds(err_table_all_rrk_mutiple(:,:,1,:),4);
errorbar(time, m, min_vec.'-m,max_vec.'-m,'LineWidth',2,MarkerSize=8)
set(gca, 'XScale','log', 'YScale','log')
hold on
loglog(time,(.5e-1.*time).^1,'--','LineWidth',1)
loglog(time,(3e-1.*time).^2,'--','LineWidth',1)
loglog(time,(.3*time).^4,'--','LineWidth',1)
yline(ref_best_error(1),"LineWidth",1.5);

legend('Location','southeast')

legendStr = [];

legendStr = [["Rand RK4","Rand RK2","Rand Euler"], "slope 1", "slope 2","slope 4","Best approximation"];

legend(legendStr)
xlabel('\Deltat')
ylabel('|| Y^{ref} - Y^{approx} ||_F')
ylim([1e-10 1])
grid on
title('alpha=1e-5')

set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
%set(gcf, 'Toolbar', 'none', 'Menu', 'none');
set(gca,'FontSize',14)


subplot(1,4,2)
title('alpha=1e-5')
loglog(time, err_table_all_prk(:,:,1).','LineWidth',2,'Marker','o',MarkerSize=8)
hold on
loglog(time, err_table_all_proj(:,:,1).','LineWidth',2,'Marker','o',MarkerSize=8)
hold on

loglog(time,(0.5e-1.*time).^1,'--','LineWidth',1)
loglog(time,(3e-1.*time).^2,'--','LineWidth',1)
loglog(time,(.3*time).^4,'--','LineWidth',1)
yline(ref_best_error(1),"LineWidth",1.5);

legend('Location','southeast')

legendStr = [];

legendStr = [["PRK 4","PRK 2","PRK 1","Proj Split"], "slope 1", "slope 2","slope 4","Best approximation"];

legend(legendStr)
xlabel('\Deltat')
ylabel('|| Y^{ref} - Y^{approx} ||_F')
ylim([1e-10 1])
grid on
title('alpha=1e-5')

set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
%set(gcf, 'Toolbar', 'none', 'Menu', 'none');
set(gca,'FontSize',14)



subplot(1,4,3)

m=mean(err_table_all_rrk_mutiple(:,:,2,:),4).';
[min_vec,max_vec] = bounds(err_table_all_rrk_mutiple(:,:,2,:),4);
errorbar(time, m, min_vec.'-m,max_vec.'-m,'LineWidth',2,MarkerSize=8)
set(gca, 'XScale','log', 'YScale','log')
hold on
loglog(time,(.5e-1.*time).^1,'--','LineWidth',1)
loglog(time,(3e-1.*time).^2,'--','LineWidth',1)
loglog(time,(.3*time).^4,'--','LineWidth',1)
yline(ref_best_error(1),"LineWidth",1.5);

legend('Location','southeast')

legendStr = [];

legendStr = [["Rand RK4","Rand RK2","Rand Euler"], "slope 1", "slope 2","slope 4","Best approximation"];

legend(legendStr)
xlabel('\Deltat')
ylabel('|| Y^{ref} - Y^{approx} ||_F')
ylim([1e-10 1])
grid on
title('alpha=1')

set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
%set(gcf, 'Toolbar', 'none', 'Menu', 'none');
set(gca,'FontSize',14)


subplot(1,4,4)
title('alpha=1')

loglog(time, err_table_all_prk(:,:,2).','LineWidth',2,'Marker','o',MarkerSize=8)
hold on
loglog(time, err_table_all_proj(:,:,2).','LineWidth',2,'Marker','o',MarkerSize=8)
hold on

loglog(time,(0.1.*time).^1,'--','LineWidth',1)
%loglog(time,(3e-1.*time).^2,'--','LineWidth',1)
%loglog(time,(.3*time).^3,'--','LineWidth',1)
yline(ref_best_error(1),"LineWidth",1.5);

legend('Location','southeast')

legendStr = [];

legendStr = [["PRK 4","PRK 2","PRK 1","Proj Split"], "slope 1","Best approximation"];

legend(legendStr)

xlabel('\Deltat')
ylabel('|| Y^{ref} - Y^{approx} ||_F')
ylim([1e-10 1])
grid on
title('alpha=1e-1')

set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
%set(gcf, 'Toolbar', 'none', 'Menu', 'none');
set(gca,'FontSize',14)
% figure()
% plot(linspace(0,0.5,length(ortho_norm)),ortho_norm)
