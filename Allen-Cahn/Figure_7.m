
% addpath and cleaning enviroment
addpath('../randRK-core')
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
A(1,N)=1;
A(N,1)=1;

M =(1/(dx.^2))*A;


%% Initial Data:

% Initial value:
Y0=zeros(K,K);

for i=1:K
    for j=1:K
        Y0(i,j)=(exp(-(tan(x(i)).^2))+exp(-(tan(x(j)).^2)))*sin(x(i))*sin(x(j))./(1+exp(abs(csc(-x(i)./2)))+exp(abs(csc(-x(j)./2))));
    end
end

eps=0.01; %eps=0.01;
H = @(Y)   (eps*(M*Y+Y*M)+Y-Y.^3);


tic
ref= odeSolver(Y0,H,0,T);
toc
F=@(Y,t) H(Y);

Time=1e-3%[2e-2,1e-2,5e-3,2.5e-3,1e-3];
err_table_all=[];
rank=2%15%[10,15,20,25,30,35,40]
sc = parallel.pool.Constant(RandStream("threefry",Seed=1234)); % set seed
rand_sol=zeros(256,256,5);
ref_sol=zeros(256,256,5);


for count=1:length(rank)
    r=rank(count)
    stream = sc.Value;        % set each worker seed
    stream.Substream =count;
    p=max(2,round(0.1*r));
    l = max(2,round(0.1*r));  %over-parametrization.
    Omega = randn(stream,n,r+p);
    Psi = randn(stream,n, r+l+p);

    X = Y0*Omega; %right-sketch
    Y = Y0'*Psi;  %left-sketch

    Y_inital = {X,Y,Omega,Psi};
    % ref = matOdeSolver(matFull(-1,Y0),F,0,T);
    errTable_randRK = [];
    flag=1;
    for dt = Time
        Y_randRK = Y_inital;
        for i=1:(T/dt)
            Y_randRK = rand_rk_4(Y_randRK,F,(i-1)*dt,i*dt,r,stream,"non constant_sketch");
            i
            if i*dt==1|| i*dt==3|| i*dt==5|| i*dt==7||i*dt==10
                ref= odeSolver(Y0,H,0,i*dt);
                ref_sol(:,:,flag)=ref;
                rand_sol(:,:,flag)=matFull(1,Y_randRK,r);
                flag=flag+1;

            end
        end

        err_randRK = norm(matFull(1,Y_randRK,r) - ref, 'fro');
        errTable_randRK = [errTable_randRK,err_randRK];
        fprintf("randRK - dt = %f, err = %e \n", dt, err_randRK);
    end
    err_table_all=[err_table_all;errTable_randRK]
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


t=[1,3,5,7,10]
tiledlayout(5,3,TileIndexing="columnmajor");
for i=1:5
    nexttile

    contourf(linspace(0, 2*pi, N),linspace(0, 2*pi, N),ref_sol(:,:,i),'LineColor','none')
    caxis([-1,1])
    if i==1
        title('Reference solution',['t=',num2str(t(i))])
    else
        title(['t=',num2str(t(i))])
    end
    set(gca,'FontSize',13)
end
cb = colorbar('southoutside');



for i=1:5
    nexttile

    contourf(linspace(0, 2*pi, N),linspace(0, 2*pi, N),rand_sol(:,:,i),'LineColor','none')
    caxis([min(min(min(rand_sol))),max(max(max(rand_sol)))])
    if i==1
        title('Randomized RK4',['t=',num2str(t(i))])
    else
        title(['t=',num2str(t(i))])
    end
    set(gca,'FontSize',13)
end
cb = colorbar('southoutside');
diff=ref_sol-rand_sol;
for i=1:5
    diff(:,:,i)=diff(:,:,i)./norm(ref_sol(:,:,i),'fro');
end
for i=1:5
    nexttile

    contourf(linspace(0, 2*pi, N),linspace(0, 2*pi, N),diff(:,:,i),'LineColor','none')
    caxis([min(min(min(diff))),max(max(max(diff)))])

    if i==1
        title('Difference',['t=',num2str(t(i))])
    else
        title(['t=',num2str(t(i))])
    end
    set(gca,'FontSize',13)
end
%caxis([min(min(min(diff))),max(max(max(diff)))])
cb = colorbar('southoutside');

