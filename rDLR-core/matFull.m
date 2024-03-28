function Y = matFull(i,x,target_rank)
    
    X = x{1};
    Y = x{2};
    Omega = x{3};
    Psi = x{4};

    tol = 100*eps(1);

    switch i
        case 1 
            %Y = X*pinv(Y'*Omega)*Y';
            [Q, ~]=qr(X,"econ");
            P=(Psi'*Q)\Y';
            [U,S,V]=svd(P,"econ");

            Y = Q*U(:,1:target_rank)*S(1:target_rank,1:target_rank)*V(:,1:target_rank)';
            
        otherwise %case 2
            %Y = X*pinv(Y'*Omega)*Y';
            %[Q,R] = qr(Psi'*X,0);
            %Y = (X*pinv(R,tol))*(Q'*Y');
    end

end