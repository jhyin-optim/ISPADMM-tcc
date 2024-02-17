function [x1,InItr,r0] = CG1(A,d,mu,x,w,vn1,y,method,opts)
global M
% B = 1/mu*A'*A + sigma*M'*M;
% M'*M = eys(n);
% b = 1/mu*A'*d + sigma*M'*w - M'* y;  % solve linear equations B*x - b=0
% the residual r(x) = B*x - b;

 sigma = opts.sigma;
 
 Ax0 = A*x;
 Mtw = M'*w;
 Mty = M'*y;
 Atd = A'*d;
 r0 = 1/mu*A'*Ax0 + sigma*x - 1/mu*Atd - sigma*Mtw + Mty;
 p0 = -r0;
 r0tr0 = r0'*r0;
 InItr = 0;
 x1 = x;

switch method
     case 'ISPADMM'
      
     sigma2 = opts.sigma2;
     alpha = opts.alpha;
     tau = (1 + alpha)/2 + 0.001;
     
     dn1 = r0; 
     normdn1 = norm(dn1);
     Mx1 = M*x1;
     normxw = norm(Mx1 - w);
     a = (1 - alpha)/(tau - alpha);
     b = 2 - a;
     c = (1 - alpha)*b;
    
     wvtd = (x1 - vn1)'*dn1;
     awd = abs(wvtd);
     an = 2*awd + normdn1^2; 
     ssn = c*sigma2*sigma*normxw^2;
    
    while an > ssn  
            
            Ap0 = A*p0;
            Cp0 = 1/mu*A'*Ap0 + sigma*p0;
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
     Mx1 = M*x1;
     normxw = norm(Mx1 - w);
     a = (1 - alpha)/(tau - alpha);
     b = 2 - a;
     c = b*(1 - alpha);
    
     wvtd = (x1 - vn1)'*dn1;
     awd = abs(wvtd);
     an = 2*awd + normdn1^2; 
     ssn = c*sigma2*sigma*normxw^2; 
     end
end


