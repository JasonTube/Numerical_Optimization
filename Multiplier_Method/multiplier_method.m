function [x, mu, lambda, output] = multiplier_method(fun, hfun, gfun, dfun, dhfun, dgfun, x0)
% PHR乘子法
% input:    fun，hfun，gfun分别是目标函数，等式约束和不等式约束
%           dfun，dhfun，dgfun分别是目标函数的梯度，约束函数Jacobi矩阵的转置，x0是迭代初值（列向量）
% output:   x是近似最优解，mu，lambda是乘子向量，output是结构变量，输出近似极小值和迭代次数等
% test:     [x, mu, lambda, output] = multiplier_method('fun', 'hfun', 'gfun', 'dfun', 'dhfun', 'dgfun', [3 3]')
max_k = 500;                                    % 最大迭代次数
sigma = 2;                                      % 罚因子
eta = 2; theta = 0.8;                           % PHR乘子法的实参数
tol = 1e-5;
inner_k = 0; outer_k = 0;                       % 内外迭代计数器
x = x0; 
he = feval(hfun, x); gi=feval(gfun, x); 
mu = 0.1 * ones(length(he),1);                  % 乘子向量mu初值
lambda = 0.1 * ones(length(gi),1);              % 乘子向量lambda初值 
beta_k = 10; beta_k_old = 10;                   % 终止条件检验值

while beta_k>tol && outer_k < max_k
    % 调用BFGS求解无约束子问题
    [x, ~, ik] = bfgs('mpsi', 'dmpsi', x0, fun, hfun, gfun, dfun, dhfun, dgfun, mu, lambda, sigma);
    inner_k = inner_k + ik;
    he = feval(hfun, x); gi = feval(gfun, x);
    beta_k = sqrt( norm(he)^2 + norm(min(gi, lambda/sigma))^2 );
    if beta_k > tol
        if outer_k >= 2 && beta_k > theta * beta_k_old
            sigma = eta * sigma;
        end
        % 更新乘子向量
        mu = mu - sigma * he;
        lambda = max(0, lambda - sigma * gi);
    end
    outer_k = outer_k + 1;
    beta_k_old = beta_k;
    x0 = x;
    
    output.fval = feval(fun, x);
    output.outer_iter = outer_k;
    output.innter_iter = inner_k;
    output.beta = beta_k;
end
