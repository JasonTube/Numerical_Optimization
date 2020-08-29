function dh = dah(mu,lamda,d,gk,Bk,delta)
n =length(d);
dh(1) = mu;
dh(2) = phi(mu,lamda,delta^2-norm(d)^2);
mh = (Bk+lamda*eye(n))*d+gk;
for i = 1:n
    dh(2+i) = mh(i);
end
dh = dh(:); % to gurantee that the output is a column vector
