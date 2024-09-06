function Y = matProjSplit_uncon(Y,F,t0,t1)
   
    dt = t1-t0;

    funK = @(K,V) F(K*V')*V;
    funL = @(L,U) F(U*L')'*U;
    funS = @(S,U,V) U'*F(U*S*V')*V;

    % variables:
    U = Y{1};
    V = Y{2};
    S = Y{3};

    % K-step
    K = U*S;
  
    %K = odeSolver(K, @(X) funK(X,V),t0,t1);
    K = rk4(K, @(X) funK(X,V),dt);
    [U1,R_1] = qr(K,0);
    
    

    %L-step:
    L = V*S';
   % L=odeSolver(L, @(X) funL(X,U),t0,t1);
    L = rk4(L, @(X) funL(X,U),dt);
    [V1,Rt_1] = qr(L,0);
    
    S=(U1'*U)*S*(V'*V1);
    %S-step
    %S=odeSolver(S, @(X) funS(X,U1,V1),t0,t1);
    S = rk4(S, @(X) funS(X,U1,V1),dt);
    %norm(Y{1}*Y{3}*Y{2}'-U1*S*V1','fro');
    Y = {U1,V1,S};
    %dt*norm(F(U1*S*V1')-calculate_pf({U1,S,V1},F,t1),'fro')
end
