function A = JacobiH(mu,lamda,d,Bk,delta)
n = length(d);
A = zeros(n+2,n+2);
pmu = -4*mu/sqrt((lamda+norm(d)^2-delta^2)^2+4*mu^2);
thetak = (lamda+norm(d)^2-delta^2)/sqrt((lamda+norm(d)^2-delta^2)^2+4*mu^2);
A = [   1,          0,          zeros(1,n)
        pmu,        1-thetak,   -2*(1+thetak)*d'
        zeros(n,1), d,          Bk+lamda*eye(n)];
