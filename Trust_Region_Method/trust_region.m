function [x,val,k] = trust_region(fun,grad_fun,hess_fun,x0)
% 信赖域-狗腿法
% input:    fun,grad_fun,hess_fun分别是目标函数,梯度和Hessian矩阵，x0是迭代初值（列向量）
% output:   x,val分别是近似最优解和最优目标函数值，k是迭代次数
% test:     [x,val,k] = trust_region('fun','grad_fun','hess_fun',[-1.2; 1])

max_k = 50;         % max iteration  
x = x0;
beta = 0.25;        % 信赖域修正准则
eta = 0.125;        % 可接受准则
delta = 5;          % delta_0
delta_bar = 20;     % max delta
tol = 1e-6;

k = 0;
while k < max_k
    grad = feval(grad_fun,x);
    B = feval(hess_fun,x);
    if norm(grad) <= tol
        break;
    end
    
    % 子问题求解
    d = dog_leg(grad, B, delta);
    
    % 信赖域修正
    delta_q = - (grad'*d + 0.5*d'*B*d);
    delta_f = feval(fun,x) - feval(fun,x+d);
    r = delta_f / delta_q;

    if r > 1 - beta && norm(d) == delta
        delta = min(2*delta, delta_bar);
    elseif  r < beta
        delta = 0.5 * delta;
    end
    
    % 可接受检测
    if r > eta
        x = x + d;
    end
    k = k + 1;
end
val = feval(fun,x);
