function [x,val,k] = CG(fun,grad_fun,x0)
% 共轭梯度法
% input:    fun,grad_fun分别是目标函数,梯度，x0是迭代初值（列向量）
% output:   x,val分别是近似最优解和最优目标函数值，k是迭代次数
% test:     [x,val,k] = CG('fun','grad_fun',[-1.2; 1])
max_k = 5000;               % max iteration
rho = 0.6; sigma = 0.4;     % armijo search params
tol = 1e-5;                 % tolerance
n = length(x0);
k = 0;
while k < max_k
    grad = feval(grad_fun, x0);
    itern = k - (n+1)*floor(k/(n+1)) + 1;
    if itern == 1
        d = - grad;
    else
        beta = (grad'*grad)/(grad_old'*grad_old);          % FR
        %beta = - (grad'*grad)/(d_old'*grad_old);           % Dixon
        %beta = (grad'*grad)/(d_old'*(grad - grad_old));    % Dai-Yuan
        %beta = (grad'*(grad - grad_old))/(d_old'*(grad - grad_old));   % Crowder-Wolfe
        %beta = (grad'*(grad - grad_old))/(d_old'*(grad - grad_old));   % HS
        %beta = (grad'*(grad - grad_old))/(grad_old'*grad_old);         % PRP
        d = -grad + beta * d_old;
        if grad'*d >= 0
            d = - grad;
        end
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
    grad_old = grad;
    d_old = d;
    k = k + 1;
end
x = x0;
val = feval(fun, x);
