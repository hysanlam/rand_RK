function Y = BUG(Y,F,t0,t1)
   
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

    %S-step
    S=(U_hat'*U)*S*(V'*V_hat);
    S=odeSolver(S, @(X) funS(X,U_hat,V_hat),t0,t1);
    %S = rk4(S, @(X) funS(X,U_hat,V_hat),dt);
    

    %Y = retruncate(U_hat,V_hat,S,1e-10);
    Y = retruncate(U_hat,V_hat,S,r);

end

function Y=retruncate(U,V,S,r)
    [U_temp,S_temp,V_temp]=svd(S,"econ");
    % d_s=diag(S_temp);
    % for output_rank = 1:length(S_temp)-1
    %     if( sqrt(sum(d_s(output_rank+1:length(S_temp)).^2)) < truncation_tol)
    %         break;
    %     end
    %     if output_rank== length(S_temp)-1
    %         output_rank=length(S_temp);
    %     end
    % end

    U=U*U_temp(:,1:r);
    S=S_temp(1:r,1:r);
    V=V*V_temp(:,1:r);

    Y = {U,V,S};

end
