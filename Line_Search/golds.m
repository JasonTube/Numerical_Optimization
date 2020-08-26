function [x,ux,k,G,E] = golds(u,a,b,delta,eps)
% 精确线搜索: 黄金分割法
% input :   u为目标函数，定义在[a,b]上，delta，eps为自变量和函数值的容许误差
% output:   x,ux为近似极小点和极小值，G的第k行记录a,p,q,b第k次的迭代值
%           E = [dx,du]为x和u的误差限
% test:     [x,ux,k,G,E] = golds(@(x)(x^2-sin(x)),0,1,1e-4,1e-5)
t = (sqrt(5)-1)/2;
h = b - a;
ua = feval(u, a);
ub = feval(u, b);
p = a+ (1-t)*h;
q = a + t*h;
up = feval(u, p);
uq = feval(u, q);
k = 1;
G(k,:) = [a, p, q, b];
while abs(ub-ua)>eps || h>delta
    if up < uq
        b = q; ub = uq;
        q = p; uq = up;
        h = b - a; p = a + (1-t)*h; up = feval(u,p);
    else
        a = q; ua = up;
        p = a; up = uq;
        h = b - a; q = a + t*h; uq = feval(u,q);
    end
    k = k + 1; G(k,:) = [a, p, q, b];
end
ds = abs(b - a);
du = abs(ub - ua);
if up < uq
    x = p; ux = up;
else
    x = q; ux = uq;
end
E = [ds, du];
