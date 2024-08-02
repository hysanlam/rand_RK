function y = projected_rk6(x0,F,t0,t1,target_rank)
    dt = t1-t0;

    k1=calculate_pf(x0,F,t0);
    k2=calculate_pf(truncate(full(x0)+dt./8*k1,target_rank),F,t0+dt./8);
    k3=calculate_pf(truncate(full(x0)+dt./18*k1+dt./9*k2,target_rank),F,t0+dt./6);
    k4=calculate_pf(truncate(full(x0)+dt./16*k1+3/16*dt*k3,target_rank),F,t0+dt./4);
    k5=calculate_pf(truncate(full(x0)+dt./4*k1-3/4*dt*k3+dt*k4,target_rank),F,t0+dt./2);
    k6=calculate_pf(truncate(full(x0)+dt*134/625*k1-333/625*dt*k3+476/625*dt*k4+98/625*dt*k5,target_rank),F,t0+dt*3/5);
    k7=calculate_pf(truncate(full(x0)+dt*(-98)/1875*k1+12/625*dt*k3+10736/13125*dt*k4-1936/1875*dt*k5+22/21*dt*k6,target_rank),F,t0+dt*4/5);
    k8=calculate_pf(truncate(full(x0)+dt*(9)/50*k1+21/25*dt*k3-2924/1925*dt*k4+74/25*dt*k5-15/7*dt*k6+15/22*dt*k7,target_rank),F,t0+dt);
   
    
    y=truncate(full(x0)+dt*(11/144*k1+256./693*k4+125./504*k6+125./528*k7+5./72*k8),target_rank);
  
    
end

function x=truncate(x,target_rank)
 [u,s,v]=svd(x);
 
 x = {u(:,1:target_rank),s(1:target_rank,1:target_rank),v(:,1:target_rank)};
end

function x=full(x0,target_rank)
 x=x0{1}*x0{2}*x0{3}';
end