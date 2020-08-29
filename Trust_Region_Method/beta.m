function bet = beta(mu,lamda,d,gk,Bk,delta,gamma)
dh = dah(mu,lamda,d,gk,Bk,delta);
bet = gamma * norm(dh)*min(1,norm(dh));
