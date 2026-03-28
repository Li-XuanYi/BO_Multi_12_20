function DB = ParEGO_Source(soh)
    %% 适配9维五阶段充电协议的源域数据生成脚本
    % 输入：soh - 电池健康状态（如0.7/1.0）
    % 输出：DB - 包含1000个9维样本的源域数据集
    % 维度说明：9维 = soc1~soc4(0.1~0.2) + i1~i5(2~6)
    
    % ========== 核心参数设置（适配9维） ==========
    dim = 9;                % 9维优化（4SOC+5电流）
    max_iter = 950;         % 迭代次数（50初始+950迭代=1000样本）
    obj_num = 3;            % 3个目标函数（充电时间/温升/寿命损耗）
    
    % 归一化上下界（0-1，实际范围在Evaluate_Bat_Source中映射）
    up = ones(1, dim);      % 1×9
    dn = zeros(1, dim);     % 1×9
    
    % 生成权重向量（保持原逻辑）
    [n_w,W] = EqualWeight(30,3);
    W = W + 1e-6;           % 避免除以0
    
    % ========== 初始化数据集（50个9维初始样本） ==========
    DB.x = (up - dn) .* rand(50, dim) + dn;  % 50×9
    DB.y = Evaluate_Bat_Source(DB.x, soh);   % 评估初始样本
    
    % ========== 主迭代（950次，最终生成1000个样本） ==========
    for iter = 1:max_iter
        % 打印迭代进度（保持原格式）
        fprintf(['当前迭代: ', num2str(iter+50), ' \n']);
        
        % ========== 构建高斯过程代理模型（保持原逻辑） ==========
        idx_w_sel = ceil(n_w * rand());
        w_sel = W(idx_w_sel, :);
        w_sel = 1./w_sel;  % 权重取倒数
        
        % 目标函数归一化（加1e-6避免除以0）
        y_min = min(DB.y);
        y_max = max(DB.y);
        y_norm = (DB.y - y_min) ./ (y_max - y_min + 1e-6);
        
        % 增广切比雪夫聚合
        y_agg = max((w_sel .* y_norm)', [], 1)' + 0.05 * sum(w_sel .* y_norm, 2);
        y_agg_norm = (y_agg - mean(y_agg)) ./ (std(y_agg) + 1e-6);
        
        % 训练GP模型
        model = fitrgp(DB.x, y_agg_norm);
              
        % ========== DE算法优化LCB（修复9维边界逻辑） ==========
        y_best = min(y_agg_norm);
        x_new = DE(model, up, dn);  % 调用修复后的DE函数
        
        % ========== 评估新解（9维） ==========
        y_new = Evaluate_Bat_Source(x_new, soh);
        
        % ========== 更新数据集 ==========
        DB.x = [DB.x; x_new];  % 追加9维新样本
        DB.y = [DB.y; y_new];
    end
    
    % ========== 非支配排序（保持原逻辑） ==========
    [FrontValue,~] = NonDominateSort(DB.y, 1);     
    Next = find(FrontValue == 1);
    PF_ = DB.y(Next, :);
    PS_ = DB.x(Next, :);
    
    % ========== 保存源域数据（文件名兼容原格式） ==========
    filename = ['DataSet_Source_ParEGO_1000_' num2str(soh * 100) '.mat'];
    save(filename, 'DB');
    fprintf(['源域数据已保存：%s\n', filename]);
end

% ========== 修复后的DE算法（适配9维，解决维度报错） ==========
function x_best = DE(dmodel, up, dn)
    [~, d] = size(dn);  % d=9（9维）
    pop_size = 30;      % DE种群大小（保持原逻辑）
    
    % 初始化30个9维个体（归一化0-1）
    x = (up - dn) .* rand(pop_size, d) + dn;  % 30×9
    
    % 初始化目标值（修复维度，改为列向量）
    y = zeros(pop_size, 1);
    for i = 1:pop_size
        y(i) = obj(x(i, :), dmodel);
    end
    
    % DE迭代优化（200次，保持原逻辑）
    for Iter = 1:200
        for i = 1:pop_size
            % 随机选择3个不同的个体
            rs = randperm(pop_size, 3);
            rj = rand(1, d);  % 9维交叉概率（和维度匹配）
            
            % ========== DE/rand/1 变异（逐元素计算） ==========
            v = x(rs(1), :) + 0.5 * (x(rs(2), :) - x(rs(3), :));
            
            % ========== 交叉（修复9维逻辑） ==========
            u = zeros(1, d);  % 初始化9维新个体
            for j = 1:d
                if rj(j) < 0.9
                    u(j) = v(j);
                else
                    u(j) = x(i, j);
                end
            end
            
            % ========== 边界修复（逐元素，解决维度不匹配） ==========
            for j = 1:d
                if u(j) > up(j)
                    u(j) = up(j);  % 超过上限则限制为上限
                elseif u(j) < dn(j)
                    u(j) = dn(j);  % 低于下限则限制为下限
                end
            end
            
            % ========== 评估新解 ==========
            y_off = obj(u, dmodel);
            
            % ========== 选择 ==========
            if y_off <= y(i)
                x(i, :) = u;
                y(i) = y_off;
            end
        end
    end
    
    % 找到最优解（9维）
    [~, idx_best] = min(y);
    x_best = x(idx_best, :);
end

% ========== 非支配排序（保持原逻辑，无需修改） ==========
function [FrontValue,MaxFront] = NonDominateSort(FunctionValue,Operation)
    if Operation == 1
        Kind = 2; 
    else
        Kind = 1;  
    end
	[N,M] = size(FunctionValue);
    
    MaxFront = 0;
    cz = zeros(1,N);
    FrontValue = zeros(1,N)+inf;
    [FunctionValue,Rank] = sortrows(FunctionValue);
    
    while (Kind==1 && sum(cz)<N) || (Kind==2 && sum(cz)<N/2) || (Kind==3 && MaxFront<1)
        MaxFront = MaxFront+1;
        d = cz;
        for i = 1 : N
            if ~d(i)
                for j = i+1 : N
                    if ~d(j)
                        k = 1;
                        for m = 2 : M
                            if FunctionValue(i,m) > FunctionValue(j,m)
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

% ========== LCB目标函数（保持原逻辑） ==========
function y = obj(x,dmodel)
    [y_pred,mse] = predict(dmodel, x);
    y = y_pred - 0.5 * mse;
end