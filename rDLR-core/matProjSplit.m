function Y = matProjSplit(Y,F,t0,t1)
   
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
    K = rk4(K, @(K) funK(K,V),dt);
    [U,S] = qr(K,0);
    
    %S-step
    S = rk4(S, @(X) -funS(X,U,V),dt);

    %L-step:
    L = V*S';
    L = rk4(L, @(L) funL(L,U),dt);
    [V,S] = qr(L,0);
    S = S';

    Y = {U,V,S};
end
