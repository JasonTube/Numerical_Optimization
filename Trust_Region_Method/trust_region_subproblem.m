function [d,val,lamda,k] = trust_region_subproblem(gk,Bk,delta)
% ��������򷽷������⣺min qk(d) = gk'*d + 0.5*d'*Bk*d, s.t. ||d||<=delta
% input:    gk��xk���ݶȣ�Bk�ǵ�k�ν���Hessian����delta�ǵ�ǰ������뾶
% output:   d,val�ֱ�������������ŵ������ֵ��lamda�ǳ���ֵ��k�ǵ�������

% test:     gk = [400; -200]; Bk = [1202 -400; -400 200]; delta = 5;
%           [d,val,lamda,k] = trust_region_subproblem(gk,Bk,delta)
maxk = 150;
n = length(gk);
gamma = 0.05;
tol = 1.0e-6;
rho = 0.6; sigma = 0.2;
mu0 = 0.05;
lamda0 = 0.05;

d0 = ones(n,1);
u0 = [mu0, zeros(1,n+1)]';
z0 = [mu0, lamda0, d0'];

z = z0;
mu = mu0;
lamda = lamda0;
d = d0;
k = 0;

while k <= maxk
    dh = dah(mu,lamda,d,gk,Bk,delta);
    if norm(dh)<tol
        break;
    end
    A = JacobiH(mu,lamda,d,Bk,delta);
    b = beta(mu,lamda,d,gk,Bk,delta,gamma)*u0 - dh;
    B = inv(A);
    dz = B*b;
    dmu = dz(1);
    dlamda = dz(2);
    dd = dz(3:n+2);
    
    m = 0; mk = 0;
    while m < 20
        dhnew = dah(mu+rho^m*dmu, lamda+rho^m*dlamda, d+rho^m*dd, gk, Bk, delta);
        if norm(dhnew) <= (1 - sigma * (1-gamma*mu0) * rho^m) * dh
            mk = m; break;
        end
        m = m + 1;
    end
    
    alpha = rho^mk;
    mu = mu + alpha * dmu;
    lamda = lamda + alpha * dlamda;
    d = d + alpha * dd;
    k = k + 1;
end
val = gk'*d + 0.5*d'*Bk*d;
