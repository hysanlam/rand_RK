function Y = matProjSplit(Y,F,t0,t1)
   
    dt = t1-t0;
    
    funK = @(K,V) F(K*V')*V;
    funL = @(L,U) F(U*L')'*U;
    funS = @(S,U,V) U'*F(U*S*V')*V;

    % variables:
    U = Y{1};
    V = Y{2};
    S = Y{3};
    %norm((eye(size(U,1))-U*U')*(F(U*S*V')*V)*V','fro')
    %dt*norm(F(U*S*V')-calculate_pf({U,S,V},F,t1),'fro')
    
    % K-step
    K = U*S;
    K = odeSolver(K, @(X) funK(X,V),t0,t1);
     %K = rk4(K, @(X) funK(X,V),dt);
    [U,S] = qr(K,0);
    
    %S-step
    S=odeSolver(S, @(X) -funS(X,U,V),t0,t1);
    %S = rk4(S, @(X) -funS(X,U,V),dt);

    %L-step:
    L = V*S';
    L=odeSolver(L, @(X) funL(X,U),t0,t1);
    %L = rk4(L, @(X) funL(X,U),dt);
    [V,S] = qr(L,0);
    S = S';
    
    Y = {U,V,S};
end
