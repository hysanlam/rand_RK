function y = randDLRA_euler(x0,F,t0,t1,target_rank,stream,constant_sketch)
%Note that Z_1=Y_1 and N_1(Z_1)=Y_1 with prob 1, we neglect the calculatation
%of N_1(Z_1)
% Here the implementation is slighly different with the notation used in
% the paper mainly on the numbering of $\Omega$ and $\Psi$, since we negect
% N_1(Z_1), we used $\Omega_1$ and $\Psi_1$ as  $\Omega_{s+1}$ and $\Psi_{s+1}$ 
    dt = t1-t0;
    X = x0{1}; 
    Y = x0{2};
    Omega = x0{3};
    Psi = x0{4};

    %select type of sketching matrix
    if constant_sketch=="non_constant_complex"
        Omega_1 = randn(stream,size(Omega,1),size(Omega,2))+1i.*randn(stream,size(Omega,1),size(Omega,2));
        Psi_1 = randn(stream,size(Psi,1),size(Psi,2))+1i.*randn(stream,size(Psi,1),size(Psi,2));
    else
        Omega_1 = randn(stream,size(Omega,1),size(Omega,2));
        Psi_1 = randn(stream,size(Psi,1),size(Psi,2));
    end

    %test=matFull(1,{X,Y,Omega,Psi},target_rank)+dt*F(matFull(1,{X,Y,Omega,Psi},target_rank),t0);
    X_1=matFull(1,{X,Y,Omega,Psi},target_rank)*Omega_1+dt*(F(matFull(1,{X,Y,Omega,Psi},target_rank),t0)*Omega_1); %sketch of Y_{i+1}
    Y_1=matFull(1,{X,Y,Omega,Psi},target_rank)'*Psi_1+dt*(F(matFull(1,{X,Y,Omega,Psi},target_rank),t0)'*Psi_1);   %sketch of Y_{i+1}

    y = {X_1,Y_1,Omega_1,Psi_1};
    %norm(test-matFull(1,{X_1,Y_1,Omega_1,Psi_1},target_rank),'fro')

end


