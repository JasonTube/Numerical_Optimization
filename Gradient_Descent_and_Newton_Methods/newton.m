function [x,val,k] = newton(fun,grad_fun,hess_fun,x0)
% ԭʼţ�ٷ�
% input:    fun,grad_fun,hess_fun�ֱ���Ŀ�꺯��,�ݶȺ�Hessian����x0�ǵ�����ֵ����������
% output:   x,val�ֱ��ǽ������Ž������Ŀ�꺯��ֵ��k�ǵ�������
% test:     [x,val,k] = newton('fun','grad_fun','hess_fun',[-1.2; 1])
max_k = 100;                % max iteration
tol = 1e-5;                 % tolerance
k = 0;
while k < max_k
    grad = feval(grad_fun, x0);
    hess = feval(hess_fun, x0);
    d = - hess \ grad;
    if norm(grad) < tol
        break;
    end
    x0 = x0 + d;
    k = k + 1;
end
x = x0;
val = feval(fun, x);
