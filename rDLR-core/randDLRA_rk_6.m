function y = randDLRA_rk_6(x0,F,t0,t1,target_rank,stream)
    dt = t1-t0;

    X = x0{1}; 
    Y = x0{2};
    Omega = x0{3};
    Psi = x0{4};

    Omega_1 = randn(stream,size(Omega,1),size(Omega,2));
    Omega_2 =randn(stream,size(Omega,1),size(Omega,2));
    Omega_3 =randn(stream,size(Omega,1),size(Omega,2));
    Omega_4 =randn(stream,size(Omega,1),size(Omega,2));
    Omega_5 =randn(stream,size(Omega,1),size(Omega,2));
    Omega_6 =randn(stream,size(Omega,1),size(Omega,2));
    Omega_7 =randn(stream,size(Omega,1),size(Omega,2));
    Omega_8 =randn(stream,size(Omega,1),size(Omega,2));

    Psi_1 = randn(stream,size(Psi,1),size(Psi,2));
    Psi_2 = randn(stream,size(Psi,1),size(Psi,2));
    Psi_3 = randn(stream,size(Psi,1),size(Psi,2));
    Psi_4 = randn(stream,size(Psi,1),size(Psi,2));
    Psi_5 = randn(stream,size(Psi,1),size(Psi,2));
    Psi_6 = randn(stream,size(Psi,1),size(Psi,2));
    Psi_7 = randn(stream,size(Psi,1),size(Psi,2));
    Psi_8 = randn(stream,size(Psi,1),size(Psi,2));
    
    k0_full=F(matFull(1,{X,Y,Omega,Psi},target_rank),t0);
    k0{1} =  k0_full*Omega_1;
    k0{2} =  k0_full'*Psi_1;
   
    temp1{1}=matFull(1,{X,Y,Omega,Psi},target_rank)*Omega_2+dt./8*k0_full*Omega_2;
    temp1{2}=matFull(1,{X,Y,Omega,Psi},target_rank)'*Psi_2+dt./8*k0_full'*Psi_2;
    
    k1_full=F(matFull(1,{temp1{1},temp1{2},Omega_2,Psi_2},target_rank),t0+dt/8 );
    k1{1}= k1_full*Omega_1;
    k1{2}=k1_full'*Psi_1;

    temp2{1}=matFull(1,{X,Y,Omega,Psi},target_rank)*Omega_3+dt./18*k0_full*Omega_3+dt./9*k1_full*Omega_3;
    temp2{2}=matFull(1,{X,Y,Omega,Psi},target_rank)'*Psi_3+dt./18*k0_full'*Psi_3+dt./9*k1_full'*Psi_3;
    
    
    k2_full= F(matFull(1,{temp2{1},temp2{2},Omega_3,Psi_3},target_rank),t0+dt/6  );
    k2{1}= k2_full*Omega_1;
    k2{2}=k2_full'*Psi_1;

    temp3{1}=matFull(1,{X,Y,Omega,Psi},target_rank)*Omega_4+dt./16*k0_full*Omega_4+dt*3/16*k2_full*Omega_4;
    temp3{2}=matFull(1,{X,Y,Omega,Psi},target_rank)'*Psi_4+dt./16*k0_full'*Psi_4+dt*3/16*k2_full'*Psi_4;

    k3_full= F(matFull(1,{temp3{1},temp3{2},Omega_4,Psi_4},target_rank),t0+dt./4 );
    k3{1}= k3_full*Omega_1;
    k3{2}=k3_full'*Psi_1;

    temp4{1}=matFull(1,{X,Y,Omega,Psi},target_rank)*Omega_5+dt./4*k0_full*Omega_5+dt*-3/4*k2_full*Omega_5+dt*k3_full*Omega_5;
    temp4{2}=matFull(1,{X,Y,Omega,Psi},target_rank)'*Psi_5+dt./4*k0_full'*Psi_5+dt*-3/4*k2_full'*Psi_5+dt*k3_full'*Psi_5;

    k4_full= F(matFull(1,{temp4{1},temp4{2},Omega_5,Psi_5},target_rank),t0+dt./2 );
    k4{1}= k4_full*Omega_1;
    k4{2}=k4_full'*Psi_1;

    temp5{1}=matFull(1,{X,Y,Omega,Psi},target_rank)*Omega_6+dt*134./625*k0_full*Omega_6+dt*-333/625*k2_full*Omega_6+476*dt./625*k3_full*Omega_6+98*dt./625*k4_full*Omega_6;
    temp5{2}=matFull(1,{X,Y,Omega,Psi},target_rank)'*Psi_6+dt*134./625*k0_full'*Psi_6+dt*-333/625*k2_full'*Psi_6+476*dt./625*k3_full'*Psi_6+98*dt./625*k4_full'*Psi_6;

    k5_full= F(matFull(1,{temp5{1},temp5{2},Omega_6,Psi_6},target_rank),t0+3*dt./5 );
    k5{1}= k5_full*Omega_1;
    k5{2}=k5_full'*Psi_1;

    temp6{1}=matFull(1,{X,Y,Omega,Psi},target_rank)*Omega_7+dt*-98./1875*k0_full*Omega_7+dt*12/625*k2_full*Omega_7+10736*dt./13125*k3_full*Omega_7-1963*dt./1875*k4_full*Omega_7+22*dt./21*k5_full*Omega_7;
    temp6{2}=matFull(1,{X,Y,Omega,Psi},target_rank)'*Psi_7+dt*-98./1875*k0_full'*Psi_7+dt*12/625*k2_full'*Psi_7+10736*dt./13125*k3_full'*Psi_7-1963*dt./1875*k4_full'*Psi_7+22*dt./21*k5_full'*Psi_7;

    k6_full= F(matFull(1,{temp6{1},temp6{2},Omega_7,Psi_7},target_rank),t0+4*dt./5 );
    k6{1}= k6_full*Omega_1;
    k6{2}=k6_full'*Psi_1;

    temp7{1}=matFull(1,{X,Y,Omega,Psi},target_rank)*Omega_8+dt*9./50*k0_full*Omega_8+dt*21/25*k2_full*Omega_8-2924*dt./1925*k3_full*Omega_8+74*dt./25*k4_full*Omega_8-15*dt./7*k5_full*Omega_8+15*dt./22*k6_full*Omega_8;
    temp7{2}=matFull(1,{X,Y,Omega,Psi},target_rank)'*Psi_8+dt*9./50*k0_full'*Psi_8+dt*21/25*k2_full'*Psi_8-2924*dt./1925*k3_full'*Psi_8+74*dt./25*k4_full'*Psi_8-15*dt./7*k5_full'*Psi_8+15*dt./22*k6_full'*Psi_8;

    k7_full= F(matFull(1,{temp7{1},temp7{2},Omega_8,Psi_8},target_rank),t0+dt );
    k7{1}= k7_full*Omega_1;
    k7{2}=k7_full'*Psi_1;

    X_1 = matFull(1,{X,Y,Omega,Psi},target_rank)*Omega_1 + (11./144*k0{1} + 256./693*k3{1}+ 125./504*k5{1}+ 125./528*k6{1}+ 5./72*k7{1})*dt;
    Y_1 = matFull(1,{X,Y,Omega,Psi},target_rank)'*Psi_1 + (11./144*k0{2} +  256./693*k3{2}+ 125./504*k5{2}+ 125./528*k6{2}+ 5./72*k7{2})*dt;

    y = {X_1,Y_1,Omega_1,Psi_1};

end


