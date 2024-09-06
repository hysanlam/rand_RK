function Y = rk4(Y,f,h)
     
     k1 = h*f(Y);
     k2 = h*f(Y + 0.5*k1);
     k3 = h*f(Y + 0.5*k2);
     k4 = h*f(Y + k3);
 
     Y = Y + (k1 + 2*k2 + 2*k3 + k4) / 6;   

end