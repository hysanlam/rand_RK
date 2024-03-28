function y = randDLRA_rk_4(x0,F,t0,t1,target_rank,stream,constant_sketch)
    dt = t1-t0;

    X = x0{1}; 
    Y = x0{2};
    Omega = x0{3};
    Psi = x0{4};
    if constant_sketch=="constant_sketch"
        Omega_1 = randn(stream,size(Omega,1),size(Omega,2));
        Omega_2 =Omega_1;
        Omega_3 =Omega_1;
        Omega_4 =Omega_1;
    
        Psi_1 = randn(stream,size(Psi,1),size(Psi,2));
        Psi_2 = Psi_1;
        Psi_3 = Psi_1;
        Psi_4 = Psi_1;
         fprintf("Constant");
        

    else 
        Omega_1 = randn(stream,size(Omega,1),size(Omega,2));
        Omega_2 =randn(stream,size(Omega,1),size(Omega,2));
        Omega_3 =randn(stream,size(Omega,1),size(Omega,2));
        Omega_4 =randn(stream,size(Omega,1),size(Omega,2));
    
        Psi_1 = randn(stream,size(Psi,1),size(Psi,2));
        Psi_2 = randn(stream,size(Psi,1),size(Psi,2));
        Psi_3 = randn(stream,size(Psi,1),size(Psi,2));
        Psi_4 = randn(stream,size(Psi,1),size(Psi,2));
    end
    k0{1} = F(matFull(1,{X,Y,Omega,Psi},target_rank),t0)*Omega_1;
    k0{2} = F(matFull(1,{X,Y,Omega,Psi},target_rank),t0)'*Psi_1;
   
    temp1{1}=matFull(1,{X,Y,Omega,Psi},target_rank)*Omega_2+dt./2*F(matFull(1,{X,Y,Omega,Psi},target_rank),t0)*Omega_2;
    temp1{2}=matFull(1,{X,Y,Omega,Psi},target_rank)'*Psi_2+dt./2*F(matFull(1,{X,Y,Omega,Psi},target_rank),t0)'*Psi_2;
    

    F_temp_1=F(matFull(1,{temp1{1},temp1{2},Omega_2,Psi_2},target_rank),t0+dt/2 );
    k1{1}= F_temp_1*Omega_1;
    k1{2}=F_temp_1'*Psi_1;

    temp2{1}=matFull(1,{X,Y,Omega,Psi},target_rank)*Omega_3+dt./2*F(matFull(1,{temp1{1},temp1{2},Omega_2,Psi_2},target_rank),t0+dt./2)*Omega_3;
    temp2{2}=matFull(1,{X,Y,Omega,Psi},target_rank)'*Psi_3+dt./2*F(matFull(1,{temp1{1},temp1{2},Omega_2,Psi_2},target_rank),t0+dt./2)'*Psi_3;
    clear temp1
    
    F_temp_2=F(matFull(1,{temp2{1},temp2{2},Omega_3,Psi_3},target_rank),t0+dt/2  );
    k2{1}= F_temp_2*Omega_1;
    k2{2}=F_temp_2'*Psi_1;

    temp3{1}=matFull(1,{X,Y,Omega,Psi},target_rank)*Omega_4+dt*F(matFull(1,{temp2{1},temp2{2},Omega_3,Psi_3},target_rank),t0+dt./2)*Omega_4;
    temp3{2}=matFull(1,{X,Y,Omega,Psi},target_rank)'*Psi_4+dt*F(matFull(1,{temp2{1},temp2{2},Omega_3,Psi_3},target_rank),t0+dt./2)'*Psi_4;
    clear temp2
    
    F_temp_3=F(matFull(1,{temp3{1},temp3{2},Omega_4,Psi_4},target_rank),t0+dt );
    k3{1}= F_temp_3*Omega_1;
    k3{2}=F_temp_3'*Psi_1;

    clear temp3;

    X_1 = matFull(1,{X,Y,Omega,Psi},target_rank)*Omega_1 + (k0{1} + 2*k1{1}+2*k2{1}+k3{1})*dt  ./ 6;
    Y_1 = matFull(1,{X,Y,Omega,Psi},target_rank)'*Psi_1 + (k0{2} + 2*k1{2}+2*k2{2}+k3{2})*dt  ./ 6;

    y = {X_1,Y_1,Omega_1,Psi_1};

end


