function [x1,InItr,r0] = CG(A,d,mu2,x,w,y,baru,vn1,method,opts)
% B = A'*A + (2*mu2 + sigmma + gamma1)*I;
% b = A'*b + sigma*w + gamma1*baru - bary;  % solve linear equations B*x - b=0
% the residual r(x) = B*x - b;

sigma = opts.sigma;
alpha = opts.alpha;
theta1 = opts.theta1;
gamma1 = opts.gamma1;
 
Atd = A'*d;
Ax = A*x;
r0 = A'*Ax + (2*mu2 + sigma + gamma1)*x - Atd - sigma*w - gamma1*baru + y ;
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
    normw = norm(x1 - w);
    wvtd = (x1 - vn1)'*dn1;
    awd = abs(wvtd);
    asn = 2*awd + sigma*normdn1^2;
    bbb =  gamma1*beta1*(1 + theta1)*normxb + beta2*sigma*(1 - alpha)*normw^2;
    
    while asn > bbb  %&& InItr<=100
            
            Ap0 = A*p0;
            Cp0 = A'*Ap0 +(2*mu2 + sigma + gamma1)*p0;
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
    normw = norm(x1 - w);
    wvtd = (x1 - vn1)'*dn1;
    awd = abs(wvtd);
    asn = 2*awd + sigma*normdn1^2;
    bbb =  gamma1*beta1*(1 + theta1)*normxb + beta2*sigma*(1 - alpha)*normw^2;  
     end
end
