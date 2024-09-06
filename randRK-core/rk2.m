function Y = rk2(Y,f,h)
     
     k1 = h*f(Y);
     k2 = h*f(Y + 0.5*k1);

     Y = Y + k2;   

end