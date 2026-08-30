function [DB,betalist] = TrOpt_exp6(init_pop,bat_type)
    tic;
    % 参数设置
    DB = load("DataSet_Source_ParEGO_1000_100.mat");
    DB = DB.DB;
    % 设置迁移的代数*
    DB.x = DB.x(1:1000,:);
    DB.y = DB.y(1:1000,:);
    DB_Source = DB;
    dim = 5;
    % 迭代次数（需要手动修改）
    max_iter = 180;
    % 设置的列表
    betalist = zeros(1,max_iter);
    up = ones(1,5);
    dn = zeros(1,5);
    % 生成权重向量（C_32^2种）
    [n_w,W] = EqualWeight(30,3);
    W = W + 0.000001;                             % 防止出现0的情况
    % 种群初始化
    DB.x = init_pop;
    % 初始评估 
    DB.y = Evaluate_Bat(DB.x,1,bat_type);
    for iter = 1:max_iter
        % 打印轮次信息
        fprintf([num2str(iter),' \n'])
        
        % 构建高斯过程代理模型
        idx_w_sel = ceil(n_w*rand());
        w_sel = W(idx_w_sel,:);
        w_sel = 1./w_sel;                         % 随机选取一个方向
        y_norm = (DB.y - min(DB.y))./(max(DB.y) - min(DB.y));
        % 增广切比雪夫方法
        y_agg = max(( w_sel.* y_norm)')' + 0.05*sum((w_sel .* y_norm),2);
        y_agg_norm = (y_agg - mean(y_agg))./std(y_agg);
        model = fitrgp(DB.x,y_agg_norm);
        
        % 构建种群分布
        [~,idx_sort] = sort(y_agg_norm);
        pop_sel = DB.x(idx_sort(1:20),:);
        mu = mean(pop_sel);
        sigma = cov(pop_sel);
        
        % 构建源种群分布
        y_source_norm = (DB_Source.y - min(DB_Source.y))./(max(DB_Source.y) - min(DB_Source.y));    % 归一化
        y_agg_source = max(( w_sel.* y_source_norm)')' + 0.05*sum((w_sel .* y_source_norm),2);      % 根据采样的权重聚集
        [~,idx_sort_source] = sort(y_agg_source);
        pop_sel_source = DB_Source.x(idx_sort_source(1:20),:) + 0.05*randn(20,dim);
        mu_source = mean(pop_sel_source);
        sigma_source = cov(pop_sel_source);
        
        % 计算KL散度与beta
        KL = multivariateGaussianKL(mu,sigma,mu_source,sigma_source);
        beta = ((max_iter-iter)/max_iter)*5/KL;
        betalist(iter) = beta;
        
        % EEI优化
        y_best = min(y_agg_norm);
        x_new = DE(model,mu_source,sigma_source,up,dn,y_best,beta);
        x_new = (x_new >= up).*up + (x_new < up).*x_new;
        x_new = (x_new <= dn).*dn + (x_new > dn).*x_new;
        y_new = Evaluate_Bat(x_new,1,bat_type);
        
        % 更新种群
        DB.x = [DB.x;x_new];
        DB.y = [DB.y;y_new];
        DB.time(iter) = toc;
    end
    % Build population distribution
    [FrontValue,~] = NonDominateSort(DB.y,1);     
    Next = find(FrontValue==1);
    PF_ = DB.y(Next,:);
    save 'DataSet_Target_EEI' DB
end

% Differential evaluation for optimizing acquisition function
function x_best = DE(dmodel,mu,C,up,dn,f_min, beta)
[~,d] = size(dn);
x = (up - dn).*rand([30,d]) + dn;
for i = 1:30
    y(i,:) = obj(x(i,:),f_min,dmodel,mu,C,beta);
end
% 迭代轮数这边修改了一下 尝试提效
for Iter = 1:20
    for i = 1:30
        rs = randperm(30,3);
        rj = rand(1,d);
        % DE/rand/1
        v(i,:) = x(rs(1),:) + 0.5*(x(rs(2),:) - x(rs(3),:));
        % Crossover
        u(i,:) = v(i,:).*(rj<0.9) + x(i,:).*(rj>=0.9);
        % Repair
        u(i,:) = (u(i,:) >= up).*up + (u(i,:) < up).*u(i,:);
        u(i,:) = (u(i,:) <= dn).*dn + (u(i,:) > dn).*u(i,:);    
        % Evaluation
        y_off(i,:) = obj(u(i,:),f_min,dmodel,mu,C,beta);
        % Selection
        if y_off(i,:) <= y(i,:)
            x(i,:) = u(i,:);
            y(i,:) = y_off(i,:);
        end
    end
end
[~,idx_best] = min(y);
x_best = x(idx_best,:);
end

% Construction of EEI
function y = obj(x,f_min,dmodel,mu,C,beta)  % obj(u(i,:),f_min,dmodel,mu,C,beta)
[~,d] = size(x);
EI = Infill_Standard_GP_EI(x, dmodel, f_min);
[y_pred,mse] = predict(dmodel,x);
P = (1/(det(C)*(2*pi)^(d/2)))*exp(-0.5*(x - mu)*(C^-1)*(x - mu)');      % 源数据建立的模型
% y = EI;% - 0.1*log(P);
y = y_pred - 0.5*mse - beta*log(P);             % y的输出公式在这里,结合了两个代理模型
end

function [FrontValue,MaxFront] = NonDominateSort(FunctionValue,Operation)
% 进行非支配排序
% 输入: FunctionValue, 待排序的种群(目标空间)，的目标函数
%       Operation,     可指定仅排序第一个面,排序前一半个体,或是排序所有的个体, 默认为排序所有的个体
% 输出: FrontValue, 排序后的每个个体所在的前沿面编号, 未排序的个体前沿面编号为inf
%       MaxFront,   排序的最大前面编号

    if Operation == 1
        Kind = 2; 
    else
        Kind = 1;  %√
    end
	[N,M] = size(FunctionValue);
    
    MaxFront = 0;
    cz = zeros(1,N);  %%记录个体是否已被分配编号
    FrontValue = zeros(1,N)+inf; %每个个体的前沿面编号
    [FunctionValue,Rank] = sortrows(FunctionValue);
    % sortrows：由小到大以行的方式进行排序，返回排序结果和检索到的数据(按相关度排序)在原始数据中的索引
    
    % 开始迭代判断每个个体的前沿面,采用改进的deductive sort，Deb非支配排序算法
    while (Kind==1 && sum(cz)<N) || (Kind==2 && sum(cz)<N/2) || (Kind==3 && MaxFront<1)
        MaxFront = MaxFront+1;
        d = cz;
        for i = 1 : N
            if ~d(i)
                for j = i+1 : N
                    if ~d(j)
                        k = 1;
                        for m = 2 : M
                            if FunctionValue(i,m) > FunctionValue(j,m)  %比较函数值，判断个体的支配关系
                                k = 0;
                                break;
                            end
                        end
                        if k == 1
                            d(j) = 1;
                        end
                    end
                end
                FrontValue(Rank(i)) = MaxFront;
                cz(i) = 1;
            end
        end
    end
end

function kl_divergence = multivariateGaussianKL(mu1, cov1, mu2, cov2)
    % Calculate the KL divergence between two multivariate Gaussian distributions.
    
    % Ensure that the covariance matrices are positive definite
    % save("DEBUG1.mat",'mu1','cov1','mu2','cov2');
    epsilon = 1e-6;
    cov1 = (cov1 + cov1') / 2;
    cov1 = cov1 + epsilon * eye(size(cov1));
    cov2 = (cov2 + cov2') / 2;
    cov2 = cov2 + epsilon * eye(size(cov2));

    chol_cov1 = chol(cov1, 'lower');
    chol_cov2 = chol(cov2, 'lower');
    
    % Calculate the determinants of the covariance matrices
    det_cov1 = prod(diag(chol_cov1))^2;
    det_cov2 = prod(diag(chol_cov2))^2;
    
    % Calculate the inverse of the covariance matrices  求逆有报警告 inv->pinv
    inv_cov1 = pinv(cov1);
    inv_cov2 = pinv(cov2);
    
    % Calculate the difference between means
    mean_diff = mu2 - mu1;
    
    % Calculate the trace term
    trace_term = trace(inv_cov2 * cov1);
    
    % Calculate the quadratic term
    quad_term = mean_diff * inv_cov2 * mean_diff';
    
    % Calculate the dimensionality
    dimension = length(mu1);
    
    % Calculate the KL divergence
    kl_divergence = 0.5 * (log(det_cov2 / det_cov1) - dimension + trace_term + quad_term);
end