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

%% 利用交替方向乘子法 (ADMM) 求解 $L1$ 正则化稀疏恢复问题
% 交替方向乘子法，即 the Alternating Direction Method of Multipliers
% （ADMM），利用 ADMM 求解 $L1$ 正则化稀疏恢复问题。
%
% 针对 $L1$正则化稀疏恢复问题
%
% $$ \min_{x,w} \frac{1}{2 /mu} \|Ax - d\|_2^2 + \|w\|_1,\quad \mathrm{s.t.}\quad Mx = w, $$
%
% 引入拉格朗日乘子 $y$ ,得到增广拉格朗日函数 
% $L_\sigma(x,w,y)=\frac{1}{2 \mu}\|Ax - d\|_2^2 + \|w\|_1 + y^\top(Mx - w) + \frac{\sigma}{2}\|Mx - w\|_2^2$。
% 在 ADMM 的每一步迭代中，交替更新 $x$, $w$，在更新 $x$( $w$) 的时候 $w$( $x$) 固定（看成常量）。
%% 初始化和迭代准备
% 函数通过优化上面给出的增广拉格朗日函数，以得到 $L1$ 正则化稀疏恢复问题的解。
%
% 输入信息： $C$, $d$, $\mu$ ，迭代初始值 $x^0$ 以及提供各参数的结构体 |opts| 。
%
% 输出信息： 迭代得到的解 $x$ 和结构体 |out| 。
%
% * |out.fvec| ：每一步迭代的 $L1$ 正则化稀疏恢复问题目标函数值
% * |out.fval| ：迭代终止时的 $L1$ 正则化稀疏恢复问题目标函数值
% * |out.tt| ：运行时间
% * |out.itr| ：迭代次数
% * |out.y| ：迭代终止时的对偶变量 $y$ 的值
% * |out.nrmC| ：约束违反度，在一定程度上反映收敛性
function [x, out] = ISPADMM(x0, mu, opts)
%%%
global A M d m n
% 迭代准备。
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

% 开始时间
tic
out.time = 0;
%% 迭代主循环
% 迭代主循环，当 (1) 达到最大迭代次数或 (2) 自变量 $x$ 的变化量小于阈值时，退出迭代循环。
while k < opts.maxit && nrmC > opts.ftol
   
    x0 = x;
    
    %非精确求解 $x$ 子问题
    [outx,InItr,dn1] = CG1(A,d,mu,x,w,vn1,y,'ISPADMM',opts);
    x = outx;
    NinItr = NinItr + InItr;
     
    %一次更新乘子 $y$
    Mx = M*x;
    bary = y + alpha * sigma * (Mx - w );
     
    w0 = w;
     
    %求解 $w$ 子问题
    c = 1/tau*Mx + 1/ts*bary - ott*w;
    w = prox(c,1 /ts);
%   c = Mx - 1 /sigma*bary;
%   w = prox(c,1 /sigma);
     
    y0 = y;

    %二次更新乘子 $y$
    y = bary + alpha * sigma * (Mx - w);
     
    %修正步
    vn1 = vn1 - dn1;

    %计算函数值和约束违反度
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
    

    % 输出每步迭代的信息。迭代步 $k$ 加一，记录当前步的函数值和约束违反度。
    % if opts.verbose
    %    fprintf('itr: %4d\tfval: %e\tfeasi:%.1e\n', k, f,nrmC);
    %    fprintf('time: %4d\tfval: %e\tfeasi:%.1e\n', k, f,nrmC);
    % end
    
    k = k + 1;
    
    out.fvec = [out.fvec; f];
    out.time = [out.time; toc];
end
%%%
% 退出循环，记录输出信息。
out.fval = f;
out.itr = k;
out.NI = NinItr;
out.time = toc;
out.nrmC = nrmC;
end
%% 辅助函数
%%%
% 函数 $h(x)=\mu\|w\|_1$ 对应的邻近算子 $\mathrm{sign}(w)\max\{|w|-\mu,0\}$。
function y = prox(w, mu)
y = max(abs(w) - mu, 0);
y = sign(w) .* y;
end
%%%
% LASSO 问题的目标函数 $f(w)=\frac{1}{2 mu}\|Ax - d\|_2^2+ \|Mx\|_1$。
function f = Func(A, d, mu, x)
global  M 
z = A * x - d;
Mx = M*x;
f = 1/(2*mu) * (z' * z) + norm(Mx, 1);
end
