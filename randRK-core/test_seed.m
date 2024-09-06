%% randomized DLRA
    % addpath and cleaning enviroment
    addpath('../rDLR-core')
    clc; clear; rng(123)
%Parameters:
n = 50; 
T = 1; %Final time.
N = 50;  

% create F
W1=normrnd(0,1,[N,N]);
W1=W1-0.5*(W1+W1');
W2=normrnd(0,1,[N,N]);
W2=W2-0.5*(W2+W2');
D=diag(10.^-(1:N));
A= @(t) expm(t.*W1)*exp(t)*D*(expm(t.*W2));
deltat=sqrt(eps)
F = @(X,t) (A(t+deltat)-A(t-deltat))./(2*deltat) ;


Y0=A(0)
Z0=Y0;
ref=A(T);



%% Randomized DLR algorithm

   time = logspace(log10(2.5e-1), log10(2.5e-4),18)%[2.5e-1,1e-1,5e-2,2.5e-2,1e-2,5e-3,2.5e-3,1e-3,5e-4,2.5e-4]; %[0.5,1e-1,1e-2,1e-3,1e-4];
   ranks = [7,9,13,17,21];

    err_table_all = []; 
    test=zeros(size(Y0,2),12,length(ranks))
    r=zeros(length(ranks),1),
    sc = parallel.pool.Constant(RandStream("threefry",Seed=123)); % set seed 

    for i=1:10
        parfor count=1:length(ranks)
             
                 stream = sc.Value;   
             
                    % set each worker seed
                stream.Substream = i*length(ranks)+count;
                r(count,1)=rand(stream),
    
              
        end
        r
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
    loglog(time,(1.*time).^2,'--','LineWidth',1)
     hold on
    loglog(time,(time).^4,'--','LineWidth',1)
        
    legend('Location','southeast')
    
    legendStr = [];
    for r = ranks
        legendStr = [legendStr, "rDLR rank = " + num2str(r)];
    end
    legendStr = [legendStr, "slope 2","slope 4"];

    legend(legendStr)
    xlabel('\Deltat')
    ylabel('|| Y^{ref} - Y^{approx} ||_F')
    ylim([1e-16 ymax])
    grid on

    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
    % Get rid of tool bar and pulldown menus that are along top of figure.
    set(gcf, 'Toolbar', 'none', 'Menu', 'none');
    set(gca,'FontSize',18)

    saveas(gcf,'randDLRA.png')
