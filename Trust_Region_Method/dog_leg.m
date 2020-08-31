function d = dog_leg(grad,B,delta)
% ���ȷ���������򷽷������⣺
% min q(d) = f(x) + grad'*d + 0.5*d'*B*d, s.t. ||d||<=delta
% input:    grad��x���ݶȣ�B�ǽ���Hessian����delta�ǵ�ǰ������뾶
% output:   d��������Ľ�

% case 1: ȫ�����Ž����������ڣ���ȫ�����Ž�
% case 2: ȫ�����Ž���ظ��ݶȷ������Žⶼ���������⣬�߸��ݶȷ���ֱ��������߽���
% case 3: ȫ�����Ž����������⣬�ظ��ݶȷ�������������ڣ����߸��ݶȷ���⣬���������Ž�֮���ߣ�ֱ��������߽���

d_B = - inv(B) * grad;                              % d_B:  ȫ�����Ž�
d_U = - ( (grad'*grad) / (grad'*B*grad) ) * grad;   % d_U:  �ظ��ݶȷ����ȫ�����Ž�
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



