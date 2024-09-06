function y = projected_rk1(x0,F,t0,t1,target_rank)
    dt = t1-t0;

    k1=calculate_pf(x0,F,t0);

    
    y=truncate(full(x0)+dt*(k1),target_rank);
  
    
end

function x=truncate(x,target_rank)
 [u,s,v]=svd(x);
 
 x = {u(:,1:target_rank),s(1:target_rank,1:target_rank),v(:,1:target_rank)};
end

function x=full(x0,target_rank)
 x=x0{1}*x0{2}*x0{3}';
end