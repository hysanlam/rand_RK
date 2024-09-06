function y = calculate_pf(x0,F,t)
    U=x0{1};
    S=x0{2};
    V=x0{3};
    I=eye(size(U,1));
    F_val=F(U*S*V',t);
    %y=(I-U*U')*(F_val*V)*V'+(U*(U'*F_val*V)*V')+U*(U'*F_val)*(I-V*V');
    y=(F_val*V)*V'+(U*(U'*F_val)-U*(U'*F_val*V)*V');

end