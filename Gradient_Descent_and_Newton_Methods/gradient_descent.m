function [x,val,k] = gradient_descent(fun,grad_fun,x0)
% �����½���
% input:    fun,dfun�ֱ���Ŀ�꺯�����ݶȣ�x0�ǵ�����ֵ����������
% output:   x,val�ֱ��ǽ������Ž������Ŀ�꺯��ֵ��k�ǵ�������
% test:     [x,val,k] = gradient_descent('fun','grad_fun',[-1.2; 1])
max_k = 5000;           % max iteration
rho = 0.5; sigma = 0.4; % armijo search params
tol = 1e-5;             % tolerance
k = 0;
while k < max_k
    grad = feval(grad_fun, x0);
    d = - grad;
    if norm(d) < tol
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
