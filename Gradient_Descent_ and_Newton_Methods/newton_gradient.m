function [x,val,k] = newton_gradient(fun,grad_fun,hess_fun,x0)
% 牛顿-最速下降混合算法
% input:    fun,grad_fun,hess_fun分别是目标函数,梯度和Hessian矩阵，x0是迭代初值（列向量）
% output:   x,val分别是近似最优解和最优目标函数值，k是迭代次数
% test:     [x,val,k] = newton_gradient('fun','grad_fun','hess_fun',[-1.2; 1])
max_k = 100;                % max iteration
rho = 0.55; sigma = 0.4;    % armijo search params
tol = 1e-5;                 % tolerance
k = 0;
while k < max_k
    grad = feval(grad_fun, x0);
    hess = feval(hess_fun, x0);
    d = - hess \ grad;
    if grad' * d >= 0
        d = - grad;
    end
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
