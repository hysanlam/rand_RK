function Y = matFull(i,x,target_rank)
    
    X = x{1};
    Y = x{2};
    Omega = x{3};
    Psi = x{4};

    tol = 100*eps(1);

    switch i
        case 1 
 
            [Q, ~]=qr(X,"econ");
            P=(Psi'*Q)\Y';
            [U,S,V]=svd(P,"econ");

            Y = Q*U(:,1:target_rank)*S(1:target_rank,1:target_rank)*V(:,1:target_rank)';

            
        otherwise %case 2
            [Q,R] = qr(Y'*Omega,0);
            [Q_X,R_X]=  qr(X*pinv(R,1e-8),"econ");
            [Q_Y,R_Y]=  qr(Y*Q,"econ");
            [U,S,V]=svd(R_X*R_Y',"econ");
            Y = Q_X*U(:,1:target_rank)*S(1:target_rank,1:target_rank)*V(:,1:target_rank)'*Q_Y';
    end

end