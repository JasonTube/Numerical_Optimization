function d = dog_leg(grad,B,delta)
% 狗腿法求解信赖域方法子问题：
% min q(d) = f(x) + grad'*d + 0.5*d'*B*d, s.t. ||d||<=delta
% input:    grad是x处梯度，B是近似Hessian矩阵，delta是当前信赖域半径
% output:   d是子问题的解

% case 1: 全局最优解在信赖域内，走全局最优解
% case 2: 全局最优解和沿负梯度方向最优解都在信赖域外，走负梯度方向直到信赖域边界上
% case 3: 全局最优解在信赖域外，沿负梯度方向解在信赖域内，先走负梯度方向解，再沿两最优解之差走，直到信赖域边界上

d_B = - inv(B) * grad;                              % d_B:  全局最优解
d_U = - ( (grad'*grad) / (grad'*B*grad) ) * grad;   % d_U:  沿负梯度方向的全局最优解
d_B_U = d_B - d_U;

if norm(d_B) <= delta           % case 1
    tau = 2;
elseif norm(d_U) >= delta       % case 2
    tau = delta / norm(d_U);
else                            % case 3
    tau = (-d_U'*d_B_U + sqrt((d_U'*d_B_U)^2 - norm(d_B_U)^2 * (norm(d_U)^2-delta^2)))/norm(d_B_U)^2 + 1;
end

if tau >= 0 && tau <= 1         % case 2
    d = tau * d_U;
elseif tau > 1 && tau <= 2      % case 1 & 3
    d = d_U + (tau - 1) * d_B_U;
end



