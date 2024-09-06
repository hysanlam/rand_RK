function y = randDLRA_odeSolver(x0,F,t0,t1)

    tspan = [t0 t1];

    X = x0{1}; 
    Y = x0{2};
    Omega = x0{3};
    Psi = x0{4};
    
    N1=size(X);
    N2=size(Y);

    y0=[X(:);Y(:)]; %vectorize X and Y and combine as one vector. 
    rfun = @(t,x) fun(x,t,F,N1,N2,Omega,Psi);

    param = odeset('RelTol', 1e-11, 'AbsTol', 1e-11);
    [~,b] = ode45(rfun, tspan, y0, param);   
    
    b = b.';
    b = b(:,end); %solution at final time.

    y = {reshape(b(1:N1(1)*N1(2)),N1),reshape(b(N1(1)*N1(2)+1:N1(1)*N1(2)+N2(1)*N2(2)),N2),Omega,Psi};
end

function z = fun(x,t,F,N1,N2,Omega,Psi)
    
    X=reshape(x(1:N1(1)*N1(2)),N1); %reshape to sketch
    Y=reshape(x(N1(1)*N1(2)+1:N1(1)*N1(2)+N2(1)*N2(2)),N2); %reshape to sketch

    mat = matFull(1,{X,Y,Omega,Psi});
    dX = F(mat)*Omega;
    dY = F(mat)'*Psi;
    z=[dX(:);dY(:)];
end