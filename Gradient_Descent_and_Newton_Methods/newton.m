function [x,val,k] = newton(fun,grad_fun,hess_fun,x0)
% 原始牛顿法
% input:    fun,grad_fun,hess_fun分别是目标函数,梯度和Hessian矩阵，x0是迭代初值（列向量）
% output:   x,val分别是近似最优解和最优目标函数值，k是迭代次数
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
