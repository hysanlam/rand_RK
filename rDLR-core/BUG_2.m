function Y=BUG_2(Y,F,t0,t1)
        dt = t1-t0;

        U0 = Y{1};
        V0 = Y{2};
        S0 = Y{3};


        r=size(S0,1);

        Y=BUG_without_truncate(Y,F,t0,t0+dt./2);
        U = Y{1};
        V = Y{2};
        S = Y{3};
        F_temp=F(U*S*V');
        [U_bar,~]=qr([U,dt*F_temp*V],0);
         [V_bar,~]=qr([V,dt*F_temp'*U],0);

        funS = @(S,U,V) U'*F(U*S*V')*V;
        S=(U_bar'*U0)*S0*(V0'*V_bar);
    
        %S-step
        S=odeSolver(S, @(X) funS(X,U_bar,V_bar),t0,t1);
        %S=rk4(S, @(X) funS(X,U_bar,V_bar),dt); 
        Y = retruncate(U_bar,V_bar,S,r);
        %norm(F(Y{1}*Y{3}*Y{2}')-calculate_pf({Y{1},Y{3},Y{2}},F,t1),'fro')

end





function Y = BUG_without_truncate(Y,F,t0,t1)
   
    dt = t1-t0;

    funK = @(K,V) F(K*V')*V;
    funL = @(L,U) F(U*L')'*U;
    funS = @(S,U,V) U'*F(U*S*V')*V;
    
    % variables:
    U = Y{1};
    V = Y{2};
    S = Y{3};
    r=size(S,1);

    % K-step
    K = U*S;
    K = odeSolver(K, @(X) funK(X,V),t0,t1);
    %K = rk4(K, @(X) funK(X,V),dt);
    [U_hat,Rt] = qr([U,K],0);

    %L-step:
    L = V*S';
    L=odeSolver(L, @(X) funL(X,U),t0,t1);
    %L = rk4(L, @(X) funL(X,U),dt);
    [V_hat,Rt_1] = qr([V,L],0);
    
    S=(U_hat'*U)*S*(V'*V_hat);
    
    %S-step
    S=odeSolver(S, @(X) funS(X,U_hat,V_hat),t0,t1);
    %S = rk4(S, @(X) funS(X,U_hat,V_hat),dt);

    Y = {U_hat,V_hat,S};
end

function Y=retruncate(U,V,S,r)
    [U_temp,S_temp,V_temp]=svd(S,"econ");
    U=U*U_temp(:,1:r);
    S=S_temp(1:r,1:r);
    V=V*V_temp(:,1:r);

    Y = {U,V,S};
end
