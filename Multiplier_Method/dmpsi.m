function dpsi = dmpsi(x, fun, hfun, gfun, dfun, dhfun, dgfun, mu, lambda, sigma)
he = feval(hfun, x); gi = feval(gfun, x);
dhe = feval(dhfun, x); dgi = feval(dgfun, x);
dpsi = feval(dfun, x) + (sigma*he - mu) * dhe + (sigma*gi - lambda) * dgi;

