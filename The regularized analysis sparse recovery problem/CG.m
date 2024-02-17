function [x1,InItr,r0] = CG(A,d,mu,x,w,y,baru,vn1,method,opts)
global M
% B = 1/mu*A'*A + sigma*M'*M + eta1*I;
% b = 1/mu*A'*d + sigma*M'*w + baru - M'*y;  % solve linear equations B*x - b=0
% the residual r(x) = B*x - b;

eta1 = opts.eta1;
sigma = opts.sigma;
alpha = opts.alpha;
theta1 = opts.theta1;

se = sigma + eta1;
 
Ax0 = A*x;
Mtw = M'*w;
Mty = M'*y;
%M'*M = eye(n);
Atd = A'*d;
r0 = 1/mu*A'*Ax0 + se*x  - 1/mu*Atd - sigma*Mtw - eta1*baru + Mty;
p0 = -r0;
r0tr0 = r0'*r0;
InItr = 0;
x1 = x;
switch method
    case 'ISPADMM_tcc'
     
    beta1 = opts.beta1;
    beta2 = opts.beta2;
    
    dn1 = r0; 
    normdn1 = norm(dn1);
    normxb = norm(x1 - baru);
    Mx1 = M*x1;
    normMw = norm(Mx1 - w);
    wvtd = (x1 - vn1)'*dn1;
    awd = abs(wvtd);
    asn = 2*awd + sigma*normdn1^2;
    btt = beta1*(1 + theta1);
    bbb = beta2*sigma*(1 - alpha)*normMw^2 + btt*normxb^2; 
    
    while asn > bbb  
            
            Ap0 = A*p0;
            Cp0 = 1/mu*A'*Ap0 + se*p0;
            p0Cp0 = p0'*Cp0;
            alphak = r0tr0/p0Cp0;
            x1 = x1 + alphak*p0;
            InItr = InItr + 1;
            r0 = r0 + alphak*Cp0;
            r0tr0_old = r0tr0;
            r0tr0 = r0'*r0;
            betak = r0tr0/r0tr0_old;
            p0 = -r0 + betak*p0;
            
    dn1 = r0; 
    normdn1 = norm(dn1);
    normxb = norm(x1 - baru);
    Mx1 = M*x1;
    normMw = norm(Mx1 - w);
    wvtd = (x1 - vn1)'*dn1;
    awd = abs(wvtd);
    asn = 2*awd + sigma*normdn1^2;
    btt = beta1*(1 + theta1);
    bbb = beta2*sigma*(1 - alpha)*normMw^2 + btt*normxb^2;  
     end
end
