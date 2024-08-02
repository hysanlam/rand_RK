function [W,V]=GN(S,U,Theta)
% output W, V such that X=W*V
    [Q,R]=qr(Theta'*S,0);
     W=S*pinv(R,10*eps(1));
     V=Q'*U; 
end