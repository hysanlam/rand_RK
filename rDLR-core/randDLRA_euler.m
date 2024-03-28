function y = randDLRA_euler(x0,F,t0,t1,target_rank,stream,constant_sketch)
    dt = t1-t0;

    X = x0{1}; 
    Y = x0{2};
    Omega = x0{3};
    Psi = x0{4};

    Omega_1 = randn(stream,size(Omega,1),size(Omega,2));
    Psi_1 = randn(stream,size(Psi,1),size(Psi,2));
   
    test=matFull(1,{X,Y,Omega,Psi},target_rank)+dt*F(matFull(1,{X,Y,Omega,Psi},target_rank),t0);
    X_1=matFull(1,{X,Y,Omega,Psi},target_rank)*Omega_1+dt*(F(matFull(1,{X,Y,Omega,Psi},target_rank),t0)*Omega_1);
    Y_1=matFull(1,{X,Y,Omega,Psi},target_rank)'*Psi_1+dt*(F(matFull(1,{X,Y,Omega,Psi},target_rank),t0)'*Psi_1);

    y = {X_1,Y_1,Omega_1,Psi_1};
    norm(test-matFull(1,{X_1,Y_1,Omega_1,Psi_1},target_rank),'fro');

end


