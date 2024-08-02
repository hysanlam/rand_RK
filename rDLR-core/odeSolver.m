function sol = odeSolver(y0, fun,  t0, t1)

    N = size(y0);

    odefun = @(t,y) F(t,y, fun, N);  
    %odefun = @(t,y) fun(t,y);
    tspan = [t0 t1];
    param = odeset('RelTol', 1e-10, 'AbsTol', 1e-10);
    [a,b] = ode45(odefun, tspan, y0(:), param);    
    b = reshape(b.', N(1), N(2),   []);
    sol = b(:,:,end);
end


function dy = F(t,y, fun, N)
    
    X = reshape(y, N);
    tmp = fun(X);
    dy = tmp(:);
end