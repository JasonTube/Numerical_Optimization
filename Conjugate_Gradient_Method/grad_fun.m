function grad = grad_fun(x)
grad = [ 400*x(1)*(x(1)^2-x(2)) + 2*(x(1)-1)
        -200*(x(1)^2-x(2)) ];
