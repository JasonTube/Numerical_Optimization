function [x,ux,k,G,E] = golds(u,a,b,delta,eps)
% ��ȷ������: �ƽ�ָ
% input :   uΪĿ�꺯����������[a,b]�ϣ�delta��epsΪ�Ա����ͺ���ֵ���������
% output:   x,uxΪ���Ƽ�С��ͼ�Сֵ��G�ĵ�k�м�¼a,p,q,b��k�εĵ���ֵ
%           E = [dx,du]Ϊx��u�������
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
