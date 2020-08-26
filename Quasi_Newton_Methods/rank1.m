function [x,val,k] = rank1(fun,grad_fun,x0)
% �Գ���1������ţ�ٷ�
% input:    fun,grad_fun�ֱ���Ŀ�꺯�����ݶȣ�x0�ǵ�����ֵ����������
% output:   x,val�ֱ��ǽ������Ž������Ŀ�꺯��ֵ��k�ǵ�������
% test:     [x,val,k] = rank1('fun','grad_fun',[-1.2; 1])
max_k = 500;                % max iteration
rho = 0.55; sigma = 0.4;    % armijo search params
tol = 1e-5;                 % tolerance
H = eye(length(x0));
k = 0;
while k < max_k
    grad = feval(grad_fun, x0);
    d = - H * grad;
    if norm(grad) < tol
        break;
    end
    m_tmp = 0; m = 0;
    while m_tmp < 20        % armijo search
        if feval(fun, x0 + rho^m_tmp * d) < ...
                feval(fun, x0) + sigma * rho^m_tmp * grad' * d
            m = m_tmp; break;
        end
        m_tmp = m_tmp + 1;
    end
    x = x0 + rho^m * d;
    s = x - x0;
    y = feval(grad_fun, x) - grad;
    H = H + (s-H*y)*(s-H*y)'/((s-H*y)'*y);
    k = k + 1;
    x0 = x;
end
val = feval(fun, x);