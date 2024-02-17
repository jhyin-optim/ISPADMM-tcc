%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%     Copyright (C) 2020 Zaiwen Wen, Haoyang Liu, Jiang Hu
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <https://www.gnu.org/licenses/>.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Change log:
%
%   2020.2.15 (Jiang Hu):
%     First version
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ���ý��淽����ӷ� (ADMM) ��� $L1$ ����ϡ��ָ�����
% ���淽����ӷ����� the Alternating Direction Method of Multipliers
% ��ADMM�������� ADMM ��� $L1$ ����ϡ��ָ����⡣
%
% ��� $L1$����ϡ��ָ�����
%
% $$ \min_{x,w} \frac{1}{2 /mu} \|Ax - d\|_2^2 + \|w\|_1,\quad \mathrm{s.t.}\quad Mx = w, $$
%
% �����������ճ��� $y$ ,�õ������������պ��� 
% $L_\sigma(x,w,y)=\frac{1}{2 \mu}\|Ax - d\|_2^2 + \|w\|_1 + y^\top(Mx - w) + \frac{\sigma}{2}\|Mx - w\|_2^2$��
% �� ADMM ��ÿһ�������У�������� $x$, $w$���ڸ��� $x$( $w$) ��ʱ�� $w$( $x$) �̶������ɳ�������
%% ��ʼ���͵���׼��
% ����ͨ���Ż���������������������պ������Եõ� $L1$ ����ϡ��ָ�����Ľ⡣
%
% ������Ϣ�� $C$, $d$, $\mu$ ��������ʼֵ $x^0$ �Լ��ṩ�������Ľṹ�� |opts| ��
%
% �����Ϣ�� �����õ��Ľ� $x$ �ͽṹ�� |out| ��
%
% * |out.fvec| ��ÿһ�������� $L1$ ����ϡ��ָ�����Ŀ�꺯��ֵ
% * |out.fval| ��������ֹʱ�� $L1$ ����ϡ��ָ�����Ŀ�꺯��ֵ
% * |out.tt| ������ʱ��
% * |out.itr| ����������
% * |out.y| ��������ֹʱ�Ķ�ż���� $y$ ��ֵ
% * |out.nrmC| ��Լ��Υ���ȣ���һ���̶��Ϸ�ӳ������
function [x, out] = ISPADMM(x0, mu, opts)
%%%
global A M d m n
% ����׼����
k = 0;
x = x0;
w = x0;
y = x0;
vn1 = x0;
NinItr = 0;
out = struct();

[m, n] = size(A);

sigma = opts.sigma;
alpha = opts.alpha;
tau = (1 + alpha)/2 + 0.001;
ts = sigma*tau;
ott = (1 - tau) /tau;

nrmC = inf;
f = Func(A, d, mu, x);
out.fvec = f;

% ��ʼʱ��
tic
out.time = 0;
%% ������ѭ��
% ������ѭ������ (1) �ﵽ������������ (2) �Ա��� $x$ �ı仯��С����ֵʱ���˳�����ѭ����
while k < opts.maxit && nrmC > opts.ftol
   
    x0 = x;
    
    %�Ǿ�ȷ��� $x$ ������
    [outx,InItr,dn1] = CG1(A,d,mu,x,w,vn1,y,'ISPADMM',opts);
    x = outx;
    NinItr = NinItr + InItr;
     
    %һ�θ��³��� $y$
    Mx = M*x;
    bary = y + alpha * sigma * (Mx - w );
     
    w0 = w;
     
    %��� $w$ ������
    c = 1/tau*Mx + 1/ts*bary - ott*w;
    w = prox(c,1 /ts);
%   c = Mx - 1 /sigma*bary;
%   w = prox(c,1 /sigma);
     
    y0 = y;

    %���θ��³��� $y$
    y = bary + alpha * sigma * (Mx - w);
     
    %������
    vn1 = vn1 - dn1;

    %���㺯��ֵ��Լ��Υ����
    f = Func(A, d, mu, x);
    Mx0 = M*x0;
    nrmxw1 = norm(Mx0 - w0);
    oa = max(norm(Mx0),norm(w0));
    ob = max(1,oa);
    nrmC1 = nrmxw1/ob;
    
    Mtww = M'*(w - w0);
    nrmxw2 = norm(sigma*Mtww);
    Mty = M'*y0;
    normMty = norm(Mty);
    oc = max(1,normMty);
    nrmC2 = nrmxw2/oc;
    nrmC = max(nrmC1,nrmC2);
    

    % ���ÿ����������Ϣ�������� $k$ ��һ����¼��ǰ���ĺ���ֵ��Լ��Υ���ȡ�
    % if opts.verbose
    %    fprintf('itr: %4d\tfval: %e\tfeasi:%.1e\n', k, f,nrmC);
    %    fprintf('time: %4d\tfval: %e\tfeasi:%.1e\n', k, f,nrmC);
    % end
    
    k = k + 1;
    
    out.fvec = [out.fvec; f];
    out.time = [out.time; toc];
end
%%%
% �˳�ѭ������¼�����Ϣ��
out.fval = f;
out.itr = k;
out.NI = NinItr;
out.time = toc;
out.nrmC = nrmC;
end
%% ��������
%%%
% ���� $h(x)=\mu\|w\|_1$ ��Ӧ���ڽ����� $\mathrm{sign}(w)\max\{|w|-\mu,0\}$��
function y = prox(w, mu)
y = max(abs(w) - mu, 0);
y = sign(w) .* y;
end
%%%
% LASSO �����Ŀ�꺯�� $f(w)=\frac{1}{2 mu}\|Ax - d\|_2^2+ \|Mx\|_1$��
function f = Func(A, d, mu, x)
global  M 
z = A * x - d;
Mx = M*x;
f = 1/(2*mu) * (z' * z) + norm(Mx, 1);
end
