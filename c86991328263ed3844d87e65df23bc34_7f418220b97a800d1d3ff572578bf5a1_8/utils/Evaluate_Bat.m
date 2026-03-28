function y = Evaluate_Bat(x,soh)
    % x: 9维
    % x(1:4) -> soc1~soc4 (优化变量)
    % x(5:9) -> i1~i5   (优化变量)

    % ========== 9维上下界 ==========
    % soc1~soc4: 0.1 ~ 0.2
    % i1~i5:   2 ~ 6
    up = [0.2, 0.2, 0.15, 0.15, 6,6,5,5,3];
    dn = [0.1, 0.1, 0.1, 0.1, 2,2,2,2,2];
    
    x = x .* (up - dn) + dn;   % 反归一化

    [m,~] = size(x);
    y = zeros(m,3);

    for i = 1:m
        soc1 = x(i,1);
        soc2 = x(i,2);
        soc3 = x(i,3);
        soc4 = x(i,4);
        soc5 = 0.8 - soc1 - soc2 - soc3 - soc4; % 自动算

        i1 = x(i,5);
        i2 = x(i,6);
        i3 = x(i,7);
        i4 = x(i,8);
        i5 = x(i,9);

        % 调用五阶段Python函数
        [m1,m2,m3,~] = mcc_5stage(soc1,soc2,soc3,soc4,soc5, i1,i2,i3,i4,i5, soh);

        y(i,1) = m1;   % 充电时间
        y(i,2) = m2;   % 温升
        y(i,3) = m3;   % 寿命损耗
    end
end