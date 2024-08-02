function [Y,output_rank] = matFull_adaptive(i,x,input_rank,truncation_tol)

    X = x{1};
    Y = x{2};
    Omega = x{3};
    Psi = x{4};    
    tol = 10*eps(1);
    
    switch i
        case 1
            %Y = X*pinv(Y'*Omega)*Y';
            [Q, ~]=qr(X,"econ");
            P=(Psi'*Q)\Y';
            [U,S,V]=svd(P,"econ");
            d_s=diag(S);
            for output_rank = 1:length(S)-1
                if( sqrt(sum(d_s(output_rank+1:length(S)).^2)) < truncation_tol)
                    break;
                end
                if output_rank== length(S)-1
                    output_rank=length(S);
                end
            end
            Y = Q*U(:,1:output_rank)*S(1:output_rank,1:output_rank)*V(:,1:output_rank)';
    
        otherwise %case 2
            %Y = X*pinv(Y'*Omega)*Y';
            %[Q,R] = qr(Psi'*X,0);
            %Y = (X*pinv(R,tol))*(Q'*Y');
    end

end