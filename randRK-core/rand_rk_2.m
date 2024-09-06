function y = rand_rk_2(x0,F,t0,t1,target_rank,stream,constant_sketch)
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

    % select type of sketching matrices.
    if constant_sketch=="constant_sketch"
        %% switch between using the same DRM
        Omega_1 = randn(stream,size(Omega,1),size(Omega,2));
        Omega_2 =Omega_1;
         
        Psi_1 = randn(stream,size(Psi,1),size(Psi,2));
        Psi_2 = Psi_1;       
         fprintf("Constant");
    elseif constant_sketch=="non_constant_complex"
        Omega_1 = randn(stream,size(Omega,1),size(Omega,2))+1i.*randn(stream,size(Omega,1),size(Omega,2));
        Omega_2 =randn(stream,size(Omega,1),size(Omega,2))+1i.*randn(stream,size(Omega,1),size(Omega,2));
       
        
        Psi_1 = randn(stream,size(Psi,1),size(Psi,2))+1i.*randn(stream,size(Psi,1),size(Psi,2));
        Psi_2 = randn(stream,size(Psi,1),size(Psi,2))+1i.*randn(stream,size(Psi,1),size(Psi,2));
    elseif constant_sketch=="constant_sketch_complex"
        Omega_1 = randn(stream,size(Omega,1),size(Omega,2))+1i.*randn(stream,size(Omega,1),size(Omega,2));
        Omega_2 =Omega_1;

        Psi_1 = randn(stream,size(Psi,1),size(Psi,2))+1i.*randn(stream,size(Psi,1),size(Psi,2));
        Psi_2 = Psi_1;       

    else 
        Omega_1 = randn(stream,size(Omega,1),size(Omega,2));
        Omega_2 =randn(stream,size(Omega,1),size(Omega,2));
       
        
        Psi_1 = randn(stream,size(Psi,1),size(Psi,2));
        Psi_2 = randn(stream,size(Psi,1),size(Psi,2));
        
    end
   
    F_temp=F(matFull(1,{X,Y,Omega,Psi},target_rank),t0); %compute F(N_1(Z_1))
    k0{1} = F_temp*Omega_1; %pre compute sketches
    k0{2} = F_temp'*Psi_1;
   
    temp{1}=matFull(1,{X,Y,Omega,Psi},target_rank)*Omega_2+dt*F_temp*Omega_2; % compute sketch of Z_2
    temp{2}=matFull(1,{X,Y,Omega,Psi},target_rank)'*Psi_2+dt*F_temp'*Psi_2; % compute sketch of Z_2
    
    F_temp=F(matFull(1,{temp{1},temp{2},Omega_2,Psi_2},target_rank),t0+dt); % compute F(N_2(Z_2))
    k1{1}= F_temp*Omega_1; % pre compute sketches
    k1{2}=F_temp'*Psi_1;
    
    X_1 = matFull(1,{X,Y,Omega,Psi},target_rank)*Omega_1 + (k0{1} + k1{1})*dt  ./ 2; %compute  Sketch of Y_{i+1}
    Y_1 = matFull(1,{X,Y,Omega,Psi},target_rank)'*Psi_1 + (k0{2} + k1{2})*dt  ./ 2;  %compute  Sketch of Y_{i+1}

    y = {X_1,Y_1,Omega_1,Psi_1};

end


