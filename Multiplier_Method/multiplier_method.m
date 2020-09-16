function [x, mu, lambda, output] = multiplier_method(fun, hfun, gfun, dfun, dhfun, dgfun, x0)
% PHR���ӷ�
% input:    fun��hfun��gfun�ֱ���Ŀ�꺯������ʽԼ���Ͳ���ʽԼ��
%           dfun��dhfun��dgfun�ֱ���Ŀ�꺯�����ݶȣ�Լ������Jacobi�����ת�ã�x0�ǵ�����ֵ����������
% output:   x�ǽ������Ž⣬mu��lambda�ǳ���������output�ǽṹ������������Ƽ�Сֵ�͵���������
% test:     [x, mu, lambda, output] = multiplier_method('fun', 'hfun', 'gfun', 'dfun', 'dhfun', 'dgfun', [3 3]')
max_k = 500;                                    % ����������
sigma = 2;                                      % ������
eta = 2; theta = 0.8;                           % PHR���ӷ���ʵ����
tol = 1e-5;
inner_k = 0; outer_k = 0;                       % �������������
x = x0; 
he = feval(hfun, x); gi=feval(gfun, x); 
mu = 0.1 * ones(length(he),1);                  % ��������mu��ֵ
lambda = 0.1 * ones(length(gi),1);              % ��������lambda��ֵ 
beta_k = 10; beta_k_old = 10;                   % ��ֹ��������ֵ

while beta_k>tol && outer_k < max_k
    % ����BFGS�����Լ��������
    [x, ~, ik] = bfgs('mpsi', 'dmpsi', x0, fun, hfun, gfun, dfun, dhfun, dgfun, mu, lambda, sigma);
    inner_k = inner_k + ik;
    he = feval(hfun, x); gi = feval(gfun, x);
    beta_k = sqrt( norm(he)^2 + norm(min(gi, lambda/sigma))^2 );
    if beta_k > tol
        if outer_k >= 2 && beta_k > theta * beta_k_old
            sigma = eta * sigma;
        end
        % ���³�������
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
