function [x1,InItr,r0] = CG1(A,d,mu2,x,w,vn1,y,method,opts)
% B = A'*A + (2*mu2 + sigmma)*I;
% b = A'*d + sigma*w - bary;  % solve linear equations B*x - b=0
% the residual r(x) = B*x - b;

 sigma = opts.sigma;
 
 Atd = A'*d;
 Ax0 = A*x;
 r0 = A'*Ax0 + (2*mu2 + sigma)*x - Atd - sigma*w + y;
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
     normxw = norm(x1 - w);
     a = (1 - alpha)/(tau - alpha);
     b = 2 - a;
     c = (1 - alpha)*b;
    
     wvtd = (x1 - vn1)'*dn1;
     awd = abs(wvtd);
     an = 2*awd + normdn1^2; 
     ssn = c*sigma2*sigma*normxw^2;
    
    while an > ssn  %&& InItr<=100
            
            Ap0 = A*p0;
            Cp0 = A'*Ap0 + (2*mu2 + sigma)*p0;
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
     normxw = norm(x1 - w);
     a = (1 - alpha)/(tau - alpha);
     b = 2 - a;
     c = (1 - alpha)*b;
    
     wvtd = (x1 - vn1)'*dn1;
     awd = abs(wvtd);
     an = 2*awd + normdn1^2; 
     ssn = c*sigma2*sigma*normxw^2; 
     end
end


