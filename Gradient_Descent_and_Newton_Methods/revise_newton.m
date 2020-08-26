function [x,val,k] = revise_newton(fun,grad_fun,hess_fun,x0)
% ����ţ�ٷ�
% input:    fun,grad_fun,hess_fun�ֱ���Ŀ�꺯��,�ݶȺ�Hessian����x0�ǵ�����ֵ����������
% output:   x,val�ֱ��ǽ������Ž������Ŀ�꺯��ֵ��k�ǵ�������
% test:     [x,val,k] = revise_newton('fun','grad_fun','hess_fun',[-1.2; 1])
max_k = 150;                % max iteration
rho = 0.55; sigma = 0.4;    % armijo search params
tol = 1e-5;                 % tolerance
tau = 0.0;
k = 0;
while k < max_k
    grad = feval(grad_fun, x0);
    mu = norm(grad)^(1+tau);
    hess = feval(hess_fun, x0);
    A = hess + mu * eye(length(x0));
    d = - A \ grad;
    if norm(grad) < tol
        break;
    end
    m_tmp = 0; m = 0;
    while m_tmp < 20        % armijo search
        if feval(fun, x0 + rho^m_tmp * d) < ...
                feval(fun, x0) + sigma * rho^m_tmp * grad' * d
            m = m_tmp; break
        end
        m_tmp = m_tmp + 1;
    end
    x0 = x0 + rho^m * d;
    k = k + 1;
end
x = x0;
val = feval(fun, x);
