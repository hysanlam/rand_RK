function y = randDLRA(x0,F,t0,t1)
    dt = t1-t0;

    X = x0{1}; 
    Y = x0{2};
    Omega = x0{3};
    Psi = x0{4};
    
    k0=dt*F(matFull(1,{X,Y,Omega,Psi}),t0);
    k1=dt*F(matFull(1,{X+0.5*k0*Omega,Y+0.5*k0'*Psi,Omega,Psi}),t0+dt./2);
    k2=dt*F(matFull(1,{X+0.5*k1*Omega,Y+0.5*k1'*Psi,Omega,Psi}),t0+dt./2);
    k3=dt*F(matFull(1,{X+k2*Omega,Y+k2'*Psi,Omega,Psi}),t1);
    
    X=X+(k0+2*k1+2*k2+k3)*Omega./6;
    Y=Y+(k0+2*k1+2*k2+k3)'*Psi./6;

    y ={X,Y,Omega,Psi};

end

