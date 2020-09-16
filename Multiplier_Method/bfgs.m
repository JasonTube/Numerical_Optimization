function [x,val,k] = bfgs(fun,grad_fun, x0, varargin)
% BFGS Method
% input:    fun,grad_fun分别是目标函数和梯度，x0是迭代初值（列向量）
% output:   x,val分别是近似最优解和最优目标函数值，k是迭代次数
% test:     [x,val,k] = bfgs('fun','grad_fun',[-1.2; 1])
max_k = 500;                % max iteration
rho = 0.55; sigma = 0.4;    % armijo search params
tol = 1e-5;                 % tolerance
B = eye(length(x0));        % B = feval('hess_fun', x0) for some cases
k = 0;
while k < max_k
    grad = feval(grad_fun, x0, varargin{:});
    d = - B \ grad;
    if norm(grad) < tol
        break;
    end
    m_tmp = 0; m = 0;
    while m_tmp < 20        % armijo search
        if feval(fun, x0 + rho^m_tmp * d, varargin{:}) < ...
                feval(fun, x0, varargin{:}) + sigma * rho^m_tmp * grad' * d
            m = m_tmp; break;
        end
        m_tmp = m_tmp + 1;
    end
    x = x0 + rho^m * d;
    s = x - x0;
    y = feval(grad_fun, x, varargin{:}) - grad;
    if  y'*s > 0
        B = B - (B*s*s'*B)/(s'*B*s) + (y*y')/(y'*s);    % BFGS
    end
    k = k + 1;
    x0 = x;
end
val = feval(fun, x, varargin{:});
