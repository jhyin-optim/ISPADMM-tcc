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
% ��ADMM�������� ADMM ��� $L1$ ����ϡ��ָ������ԭ���⡣
%
% ��� $L1$����ϡ��ָ�����
%
% $$ \min_{x,w} \frac{1}{2 /mu} \|Ax - d\|_2^2 + \|w\|_1,\quad \mathrm{s.t.}\quad Mx = w, $$
%
% �����������ճ��� $y$ ,�õ������������պ��� 
% $L_\sigma(x,w,y)=\frac{1}{2 \mu}\|Ax - d\|_2^2 + \|w\|_1 + y^\top(Mx - w) + \frac{\sigma}{2}\|Mx - w\|_2^2$��
% �� ADMM ��ÿһ�������У�������� $x$, $w$���ڸ��� $x$( $w$) ��ʱ�� $w$( $x$) �̶������ɳ�������
%% ��ʼ���͵���׼��
% ����ͨ���Ż���������������������պ������Եõ� $L1$����ϡ��ָ�����Ľ⡣
%
% ������Ϣ�� $A$, $d$, $\mu$ ��������ʼֵ $x^0$ �Լ��ṩ�������Ľṹ�� |opts| ��
%
% �����Ϣ�� �����õ��Ľ� $x$ �ͽṹ�� |out| ��
%
% * |out.fvec| ��ÿһ�������� $L1$ ����ϡ��ָ�����Ŀ�꺯��ֵ
% * |out.fval| ��������ֹʱ�� $L1$ ����ϡ��ָ�����Ŀ�꺯��ֵ
% * |out.tt| ������ʱ��
% * |out.itr| ����������
% * |out.y| ��������ֹʱ�Ķ�ż���� $y$ ��ֵ
% * |out.nrmC| ��Լ��Υ���ȣ���һ���̶��Ϸ�ӳ������
function [x, out] = ISPADMM_tcc(x0, mu1, mu2, opts)
%%%
global A d m n
% ����׼����
k = 0;
x = x0;
w = x0;
y = x0;
u = x0;
v = x0;
vn1 = x0;
NinItr = 0;
out = struct();

[m, n] = size(A);

sigma = opts.sigma;
alpha = opts.alpha;
theta1 = opts.theta1;
theta2 = opts.theta2;
gamma2 = opts.gamma2;

%���ű��
sg = sigma + gamma2;

%%%
nrmC = inf;
f = Func(A, d, mu1, mu2, x);
out.fvec = f;

% ��ʼʱ��
tic
out.time = 0;
%% ������ѭ��
% ������ѭ������ (1) �ﵽ������������ (2) �Ա��� $x$ �ı仯��С����ֵʱ���˳�����ѭ����
while k < opts.maxit && nrmC > opts.ftol
    
    %x0 = x;
    
    % $u$-͹��ϼ���
    baru = (1 - theta1)*x + theta1*u;
    
    %�Ǿ�ȷ��� $x$ ������
    [x1,InItr,dn1] = CG(A,d,mu2,x,w,y,baru,vn1,'ISPADMM_tcc',opts);
    x = x1;
    NinItr = NinItr + InItr;
    
    %һ�θ��³��� $y$
    bary = y + alpha * sigma * (x - w );
    
    % $v$-͹��ϼ���
    barv = (1 - theta2)* w + theta2 * v;
    
    %w0 = w;
    
    %��� $w$ ������
    c =  gamma2/sg * barv + sigma/sg * x + 1/sg * bary;
    w = prox(c,mu1/sg);
    
    %y0 = y;
    
    %���θ��³��� $y$
    y = bary + alpha * sigma * (x - w);

    %������
    vn1 = vn1 - sigma * dn1;

    %���㺯��ֵ��Լ��Υ����
    f = Func(A, d, mu1, mu2, x);
    
    nrmxw1 = norm(x - w);
    oa = max(norm(x),norm(w));
    ob = max(1,oa);
    nrmC1 = nrmxw1/ob; 
   
    gwb = gamma2*(w - barv);
    %gwb = sigma*(w - barv);
    nrmxw2 = norm(gwb);
    normy = norm(y);
    oc = max(1,normy);
    nrmC2 = nrmxw2/oc;
    
    nrmC = max(nrmC1,nrmC2);
    % nrmxw1 = norm(x0 - w0);
    % oa = max(norm(x0),norm(w0));
    % ob = max(1,oa);
    % nrmC1 = nrmxw1/ob; 
    %    
    % gwb = gamma2*(w - barv);
    % nrmxw2 = norm(gwb);
    % normy0 = norm(y0);
    % oc = max(1,normy0);
    % nrmC2 = nrmxw2/oc;
    
    % ���ÿ����������Ϣ�������� $k$ ��һ����¼��ǰ���ĺ���ֵ��Լ��Υ���ȡ�
    % if opts.verbose
    %    fprintf('itr: %4d\tfval: %e\tfeasi:%.1e\n', k, f,nrmC);
    %    fprintf('time: %4d\tfval: %e\tfeasi:%.1e\n', k, f,nrmC);
    % end
    
    k = k + 1;
    u = baru;
    v = barv;
    
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
% ���� $h(w)=\mu1\|w\|_1$ ��Ӧ���ڽ����� $\mathrm{sign}(w)\max\{|w|-\mu1,0\}$��
function y = prox(w, mu1)
y = max(abs(w) - mu1, 0);
y = sign(w) .* y;
end
%%%
% $L1$����ϡ��ָ������Ŀ�꺯�� $f(x)=\frac{1}{2}\|Ax - b\|_2^2+ \mu1\|x\|_1+\mu2\|x\|^2$��
function f = Func(A, d, mu1, mu2, x)
z = A * x - d;
f = 0.5 * (z' * z) + mu1*norm(x, 1) + mu2* norm(x, 2)^2;
end
