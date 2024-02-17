
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


%% ʵ�������淽����ӷ���ADMM���� $L1$����ϡ��ָ�����
% ����$L1$����ϡ��ָ����� 
% 
% $$\min_x \frac{1}{2 \mu}\|Ax - d\|_2^2 + \|Mx\|_1.$$
% 
% ���ȿ������� ADMM ���ԭ���⣺����ת��Ϊ ADMM ��׼����
% $$ 
% \begin{array}{rl}
% \displaystyle\min_{x,w} & \hspace{-0.5em}\frac{1}{2 /mu}\|Ax - d\|^2_2+\|w\|_1,\\
% \displaystyle\mathrm{s.t.} & \hspace{-0.5em} Mx = w,
% \end{array} 
% $$
%% ���� $L1$����ϡ��ָ�����
% �趨������ӡ�
clear;
seed = 97006855;
ss = RandStream('mt19937ar','Seed',seed);
RandStream.setGlobalStream(ss);
%%
% ���� $L1$����ϡ��ָ��Ż�����
% $$\displaystyle\min_x \frac{1}{2 /mu}\|Ax - d\|_2^2+\|Mx\|_1.$$
% ��������ľ��� $A$ ������ $u$ ��ʹ�� $b=Au$�� ����ϵ�� $\mu=10^{-3}$�� ���������ʼ�㡣
% ������Ľṹ�� |opts| �ж�ȡ�������ȡĬ�ϲ�����
% * |opts.maxit| ������������
% * |opts.ftol| ����Ժ���ֵ��ͣ��׼�򣬵��������ε�������ֵ֮��С�ڸ�ֵʱ��Ϊ����������
% * |opts.gtol| ����� $y$ ���ݶȵ�ͣ��׼�򣬵���ǰ�����ݶȷ���С�ڸ�ֵʱ��Ϊ����������
% * |opts.sigma| ��������������ϵ��
% * |opts.alpha| �� $x$ ���µĲ���
% * |opts.verbose| ����Ϊ 0 ʱ���ÿ��������Ϣ���������

global A d m n

%data = [300 600;600 1200;800 1600;1000 2000;1500 3000;2000 4000;3000 6000;4000 8000;5000 10000;6000 12000];
data = [100 256;200 512;400 1024;800 2048;1000 2048;2000 4096;3000 8192];
[row,~] = size(data);
fid = fopen('mytext.txt','w');

%ISPADMM_tcc
opts.verbose = 0;
opts.maxit = 5000;
opts.sigma = 8;
opts.theta1 = 0.1;
opts.theta2 = 0.2;
opts.gamma1 = 0.01;
opts.gamma2 = 0.02;
opts.alpha = 0.3;
opts.beta1 = 0.98;%0.01
opts.beta2 = 0.99;
opts.ftol = 2e-3; 
opts.gtol = 2e-4;

%IPSADMM
opts.verbose = 0;
opts.maxit = 5000;
opts.sigma = 8;
opts.alpha = 0.3;
opts.sigma2 = 0.99;
opts.ftol = 2e-3; 
opts.gtol = 2e-4;

%ģ�Ͳ���
%mu = 0.01;
mu1 = 0.01;%0.1*norm(A'*b,inf);
mu2 = 0.02;%0.2*norm(A'*b,inf);

for index = 1:row
    m = data(index,1);
    n = data(index,2);   
    progress_r = [];
    for repeats=1:7
        p = 100/n;
        %p = 0.1*sqrt(n);
        x0 = zeros(n,1);%randn(n,1);
        %w0 = zeros(m,1);%sprandn(n,1,p);
        H = hadamard(n);
        selected_rows = randperm(m, m); % ���ѡ��m��
        selected_columns = randperm(n, n); % ���ѡ��n��
        A = H(selected_rows, selected_columns);
        %M = A*spdiags(1./sqrt(sum(A.^2))',0,n,n); %normlize columns 
        %G = randn (n, n);
         
       %���� $G$ �� QR �ֽ�

%         R = zeros(n);
%         Q = zeros(n);
% 
%       for k = 1 : n
%         R(k,k) = norm(G(:,k)); % ����R�ĶԽ���Ԫ��
%         Q(:,k) = G(:,k) / R(k,k); % G������������������׼�����õ���������Q
%            for i = k + 1 : n
%         R(k,i) = G(:,i)' * Q(:,k); % ����R�������ǲ���
%         G(:,i) = G(:,i) - R(k,i) .* Q(:,k); % ���¾���G��˹����������ʽ
%            end
%       end
%         M = Q;

        %u = M*w0; 
        d = A * x0 + sqrt(0.01)*randn(m,1);%

      
        disp('Starting ISPADMM_tcc')  
        [x, out1] = ISPADMM_tcc(x0, mu1, mu2, opts);
        
        disp('Starting ISPADMM')
        [x, out2] = ISPADMM(x0, mu1, mu2, opts);
        
        progress_r = [progress_r; out1.itr out1.NI out1.time out1.fval out1.nrmC,...
                                  out2.itr out2.NI out2.time out2.fval out2.nrmC];      
    end
    fprintf(fid,'%d & %d  & %.0f/%.0f/%.3f/%.3e/%.3e & %.0f/%.0f/%.3f/%.3e/%.3e \\\\\n', m,n,mean(progress_r));         
end
fclose(fid);    

% fprintf('k1=%d\t nrmC1= %.2e\t fvec1= %.3f\n', k1,out1.nrmC, out1.fvec(end));
% fprintf('k2=%d\t nrmC2= %.2e\t fvec2=%.3f\n', k2,out2.nrmC, out2.fvec(end));
% fprintf('k3=%d\t nrmC3=%.2e\t fvec3=%.3f\n', k3,out3.nrmC, out3.fvec(end));
% fprintf('k4=%d\t nrmC4= %.2e\t fvec4=%.3f\n', k4,out4.nrmC, out4.fvec(end));
%% ������ӻ�
% ��ÿһ����Ŀ�꺯��ֵ�����ź���ֵ����������п��ӻ���
% fig = figure;
% semilogy(0:k1-1, data1, '-', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
% hold on
% semilogy(0:k2-1, data2, '-.','Color',[0.99 0.1 0.2], 'LineWidth',1.5);
% legend('IPSADMM_tcc��ԭ����');
% ylabel('$(\Phi(x^k) - \Phi^*)/\Phi^*$', 'fontsize', 14, 'interpreter', 'latex');
% xlabel('������');
% fig=figure;
% semilogy(out1.time, out1.fvec, '-', 'Color',[0.2 0.1 0.99], 'LineWidth',2.5);
% hold on
% semilogy(out2.time, out2.fvec, '-.','Color',[0.99 0.1 0.2], 'LineWidth',2.5);
% hold on
% semilogy(out3.time, out3.fvec, '-.','Color',[0.1 0.99 0.2], 'LineWidth',2.5);
% hold on
% semilogy(out4.time, out4.fvec, '-.','Color',[0.2 0.1 0.99], 'LineWidth',2.5);
% legend('SCPRSM','GrpADMM','IPPRSM-cc');'IPSADMM_tcc');
% ylabel('Objective Value ','fontsize', 14, 'interpreter', 'latex');%$(\Phi(x^k) - \Phi^*)/\Phi^*$',
% xlabel('CPU time (in seconds)');
% print(fig, '-depsc','admm.eps');

% fig = figure;
% semilogy(0:k1-1, out1.fvec, '-', 'Color',[0.2 0.1 0.99], 'LineWidth',2);
% hold on
% semilogy(0:k2-1, out2.fvec, '-.','Color',[0.99 0.1 0.2], 'LineWidth',2);
% hold on
% semilogy(0:k3-1, out3.fvec, '-.','Color',[0.1 0.99 0.2], 'LineWidth',2);
% hold on
% %semilogy(0:k4-1, out4.fvec, ':','Color',[0.1 0.3 0.99], 'LineWidth',2);
% legend('SCPRSM','GrpADMM','IPPRSM-cc','IPSADMM_tcc');
% ylabel('Objective Value ', 'fontsize', 14, 'interpreter', 'latex');%'$(\Phi(x^k) - \Phi^*)/\Phi^*$',
% xlabel('iteration');
% print(fig, '-depsc','admm.eps');
