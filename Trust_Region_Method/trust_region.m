function [x,val,k] = trust_region(fun,grad_fun,hess_fun,x0)
% ������-���ȷ�
% input:    fun,grad_fun,hess_fun�ֱ���Ŀ�꺯��,�ݶȺ�Hessian����x0�ǵ�����ֵ����������
% output:   x,val�ֱ��ǽ������Ž������Ŀ�꺯��ֵ��k�ǵ�������
% test:     [x,val,k] = trust_region('fun','grad_fun','hess_fun',[-1.2; 1])

max_k = 50;         % max iteration  
x = x0;
beta = 0.25;        % ����������׼��
eta = 0.125;        % �ɽ���׼��
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
    
    % ���������
    d = dog_leg(grad, B, delta);
    
    % ����������
    delta_q = - (grad'*d + 0.5*d'*B*d);
    delta_f = feval(fun,x) - feval(fun,x+d);
    r = delta_f / delta_q;

    if r > 1 - beta && norm(d) == delta
        delta = min(2*delta, delta_bar);
    elseif  r < beta
        delta = 0.5 * delta;
    end
    
    % �ɽ��ܼ��
    if r > eta
        x = x + d;
    end
    k = k + 1;
end
val = feval(fun,x);
