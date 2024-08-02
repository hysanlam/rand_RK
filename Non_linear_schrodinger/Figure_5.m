%% Compare with PRK
numer_trials=10;
K=100;
N=K
n=K
T=5;
M =diag(1*ones(1,K-1),1) + diag(1*ones(1,K-1),-1);

% change case here
%% case 1:
alpha=[3e-4,3e-1]%1%1e-1%1e-5;
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

%make sure the orthogonal part are the same across platform, avoid
%rounding error gives different orthogonal parts:
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

test_method_rrk=["randDLRA_rk_4","randDLRA_rk_2","randDLRA_euler"]


ref_best_error=[];
sc = parallel.pool.Constant(RandStream("threefry",Seed=1234)); % set seed
err_table_all_rrk_mutiple=zeros(length(test_method_rrk),length(time),length(alpha),numer_trials);
for trial=1:numer_trials
    for alpha_count=1:length(alpha)
        err_table_all_rrk_temp=[];

        H = @(Y)   1i*(0.5*(M*Y+Y*M)+alpha(alpha_count)*(abs(Y).^2).*Y);
        F=@(Y,t) H(Y);
        ref = odeSolver(Y0,F,0,T);
        r = 30; %[2,4,8,16,32]
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

                    Y_randDLRA = fun(Y_randDLRA,F,(i-1)*dt,i*dt,r,stream,"non_constant_complex");
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
err_table_all_prk=zeros(length(test_method_prk),length(time),length(alpha));;
ortho_norm_all=[];
r = 30; %[2,4,8,16,32]

parfor alpha_count=1:length(alpha)
    err_table_all_prk_temp=[]

    H = @(Y)   1i*(0.5*(M*Y+Y*M)+alpha(alpha_count)*(abs(Y).^2).*Y);
    F=@(Y,t) H(Y);

    ref = odeSolver(Y0,F,0,T);


    for  funname=1:length(test_method_prk)

        [U,S,V]=svd(Y0);
        Y_inital = {U(:,1:r),S(1:r,1:r),V(:,1:r)};

        % ref = matOdeSolver(matFull(-1,Y0),F,0,T);
        errTable_prk = [];
        ortho_norm=[];
        for dt = time

            Y_projected= Y_inital;
            maxT = round(T/dt);

            for i=1:maxT
                fun=str2func(test_method_prk(funname));
                if test_method_prk(funname)=="projected_rk2" & dt== 2.5e-4
                    temp=norm(F(Y_projected{1}*Y_projected{2}*Y_projected{3}',i*dt)-calculate_pf(Y_projected,F,i*dt),"fro");
                    ortho_norm=[ortho_norm,temp];
                end
                %fprintf("r = %d, t = %f \n", r, i*dt);

                Y_projected = fun(Y_projected,F,(i-1)*dt,i*dt,r);

            end
            ortho_norm_all=[ortho_norm_all;ortho_norm];
            %ref = odeSolver(Y0,F,0,i*dt);
            err_prk = norm(Y_projected{1}*Y_projected{2}*Y_projected{3}' - ref, 'fro');
            errTable_prk = [errTable_prk,err_prk];
            fprintf("PRK - dt = %f, err = %e \n", dt, err_prk);
        end

        err_table_all_prk_temp=[ err_table_all_prk_temp;errTable_prk];
    end
    err_table_all_prk(:,:,alpha_count)= err_table_all_prk_temp
end

%%%%%%%%%%% Projector splitting %%%%%%%%%%%
test_method=["matProjSplit"]
err_table_all_proj=zeros(length(test_method),length(time),length(alpha));
ortho_norm=[];
ref_best_error=[];
parfor alpha_count=1:length(alpha)
    err_table_all_proj_temp=[];
    H = @(Y)   1i*(0.5*(M*Y+Y*M)+alpha(alpha_count)*(abs(Y).^2).*Y);
    F=@(Y,t) H(Y);

    ref = odeSolver(Y0,F,0,T);
    [U_ref,sg_ref,V_ref] = svd(ref);
    ref_best_error=[ref_best_error,norm(ref-U_ref(:,1:r)*sg_ref(1:r,1:r)*V_ref(:,1:r)',"fro")];


    for funname=1:length(test_method)


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

%% Plotting


subplot(1,4,1)
m=mean(err_table_all_rrk_mutiple(:,:,1,:),4).';
[min_vec,max_vec] = bounds(err_table_all_rrk_mutiple(:,:,1,:),4);
errorbar(time, m, min_vec.'-m,max_vec.'-m,'LineWidth',2,MarkerSize=8)
set(gca, 'XScale','log', 'YScale','log')
hold on
loglog(time,(100*time).^1,'--','LineWidth',1)
loglog(time,(5*time).^2,'--','LineWidth',1)
loglog(time,(1*time).^4,'--','LineWidth',1)
yline(ref_best_error(1),"LineWidth",1.5);

legend('Location','southeast')

legendStr = [];

legendStr = [["Rand RK4","Rand RK2","Rand Euler"], "slope 1", "slope 2","slope 4","Best approximation"];

legend(legendStr)
xlabel('\Deltat')
ylabel('|| Y^{ref} - Y^{approx} ||_F')
ylim([1e-10 1])
grid on
title('alpha=1e-2')

set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
%set(gcf, 'Toolbar', 'none', 'Menu', 'none');
set(gca,'FontSize',12)


subplot(1,4,2)
title('alpha=5e-2')

loglog(time, err_table_all_proj(:,:,1).','LineWidth',2,'Marker','o',MarkerSize=8)
hold on
loglog(time, err_table_all_prk(:,:,1).','LineWidth',2,'Marker','o',MarkerSize=8)
loglog(time,(100*time).^1,'--','LineWidth',1)
loglog(time,(5*time).^2,'--','LineWidth',1)
loglog(time,(3*time).^4,'--','LineWidth',1)
yline(ref_best_error(1),"LineWidth",1.5);

legend('Location','southeast')

legendStr = [];

legendStr = [["Proj Split","PRK 4","PRK 2","PRK 1"], "slope 1", "slope 2","slope 4","Best approximation"];


legend(legendStr)
xlabel('\Deltat')
ylabel('|| Y^{ref} - Y^{approx} ||_F')
ylim([1e-10 1])
grid on
title('alpha=1e-1')
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
%set(gcf, 'Toolbar', 'none', 'Menu', 'none');
set(gca,'FontSize',12)


subplot(1,4,3)
m=mean(err_table_all_rrk_mutiple(:,:,2,:),4).';
[min_vec,max_vec] = bounds(err_table_all_rrk_mutiple(:,:,2,:),4);
errorbar(time, m, min_vec.'-m,max_vec.'-m,'LineWidth',2,MarkerSize=8)
set(gca, 'XScale','log', 'YScale','log')
hold on
loglog(time,(100.*time).^1,'--','LineWidth',1)
loglog(time,(7.*time).^2,'--','LineWidth',1)
loglog(time,(3*time).^3,'--','LineWidth',1)
loglog(time,(2*time).^4,'--','LineWidth',1)
yline(ref_best_error(2),"LineWidth",1.5);

legend('Location','southeast')

legendStr = [];

legendStr = [["Rand RK4","Rand RK2","Rand Euler"], "slope 1", "slope 2","slope 4","Best approximation"];


legend(legendStr)
xlabel('\Deltat')
ylabel('|| Y^{ref} - Y^{approx} ||_F')
ylim([1e-10 1])
grid on
title('alpha=4e-1')
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
%set(gcf, 'Toolbar', 'none', 'Menu', 'none');
set(gca,'FontSize',12)
subplot(1,4,4)

loglog(time, err_table_all_proj(:,:,2).','LineWidth',2,'Marker','o',MarkerSize=8)
hold on
loglog(time, err_table_all_prk(:,:,2).','LineWidth',2,'Marker','o',MarkerSize=8)
loglog(time,(100.*time).^1,'--','LineWidth',1)
loglog(time,(5.*time).^2,'--','LineWidth',1)


yline(ref_best_error(2),"LineWidth",1.5);

legend('Location','southeast')

legendStr = [];

legendStr = [["Proj Split","PRK 4","PRK 2","PRK 1"], "slope 1","slope 2","Best approximation"];


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


% figure()
% plot(linspace(0,0.5,length(ortho_norm)),ortho_norm)