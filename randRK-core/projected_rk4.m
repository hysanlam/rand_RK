function y = projected_rk4(x0,F,t0,t1,target_rank)
    dt = t1-t0;

    k1=calculate_pf(x0,F,t0);
    k2=calculate_pf(truncate(full(x0)+dt./2*k1,target_rank),F,t0+dt./2);
    k3=calculate_pf(truncate(full(x0)+dt./2*k2,target_rank),F,t0+dt./2);
    k4=calculate_pf(truncate(full(x0)+dt*k3,target_rank),F,t0+dt);
    y=truncate(full(x0)+dt./6*(k1+2*k2+2*k3+k4),target_rank);
  
    
end

function x=truncate(x,target_rank)
 [u,s,v]=svd(x);
 
 x = {u(:,1:target_rank),s(1:target_rank,1:target_rank),v(:,1:target_rank)};
end

function x=full(x0,target_rank)
 x=x0{1}*x0{2}*x0{3}';
end