function y = Evaluate_Bat_Source(x, soh)
    %% 源域数据评估函数（9维五阶段充电协议）
    % x: 输入种群矩阵 (m×9)，归一化0-1
    %    x(:,1-4) → soc1~soc4（反归一化到0.1~0.2）
    %    x(:,5-9) → i1~i5（反归一化到2~6A）
    % soh: 电池健康状态
    
    % ========== 9维参数反归一化映射 ==========
    % soc1~soc4: 0.1 ~ 0.2
    soc_up = [0.2, 0.2, 0.15, 0.15];
    soc_dn = [0.1, 0.1, 0.1, 0.1];
    
    % i1~i5: 2 ~ 6A
    i_up = [6, 6, 5, 5, 3];
    i_dn = [2, 2, 2, 2, 2];
    
    % 合并上下界（9维）
    up = [soc_up, i_up];    % 1×9
    dn = [soc_dn, i_dn];    % 1×9
    
    % 反归一化到实际范围
    x = x .* (up - dn) + dn;
    
    [m, ~] = size(x);
    y = zeros(m, 3);  % 3个目标函数
    
    % ========== 逐样本评估 ==========
    for i = 1:m
        % 提取9维参数
        soc1 = x(i, 1);
        soc2 = x(i, 2);
        soc3 = x(i, 3);
        soc4 = x(i, 4);
        soc5 = 0.8 - soc1 - soc2 - soc3 - soc4;  % 自动计算soc5
        
        i1 = x(i, 5);
        i2 = x(i, 6);
        i3 = x(i, 7);
        i4 = x(i, 8);
        i5 = x(i, 9);
        
        % 调用五阶段MCC函数（适配Python的mcc_5stage）
        [m1, m2, m3, ~] = mcc_5stage(soc1, soc2, soc3, soc4, soc5, i1, i2, i3, i4, i5, soh);
        
        % 赋值目标函数
        y(i, 1) = m1;  % 充电时间（分钟）
        y(i, 2) = m2;  % 温升（℃）
        y(i, 3) = m3;  % 寿命损耗（%）
    end
end