function x_best = DE(dmodel,up,dn)
    [~,d] = size(dn);  % d=9（9维）
    x = rand(30,d).*(up-dn)+dn;  % 30个个体，9维
    
    % 初始化目标值
    y = zeros(30,1);
    for i = 1:30
        y(i) = obj(x(i,:),dmodel);
    end
    
    % DE迭代优化
    for Iter = 1:200
        for i = 1:30
            rs = randperm(30,3);
            rj = rand(1,d);  % 9维的随机数（和维度匹配）
            
            % DE/rand/1 变异
            v = x(rs(1),:) + 0.5*(x(rs(2),:)-x(rs(3),:));
            
            % 交叉（逐元素判断，适配9维）
            u = zeros(1,d);  % 先初始化9维的u
            for j = 1:d      % 逐元素处理交叉
                if rj(j) < 0.9
                    u(j) = v(j);
                else
                    u(j) = x(i,j);
                end
            end
            
            % ========== 修复：逐元素边界限制（核心改这里） ==========
            % 代替原来的 u(u>up)=up; 避免维度不匹配
            for j = 1:d
                if u(j) > up(j)  % 第j维超过上限，限制为上限
                    u(j) = up(j);
                elseif u(j) < dn(j)  % 第j维低于下限，限制为下限
                    u(j) = dn(j);
                end
            end
            
            % 评估新解
            y_off = obj(u,dmodel);
            
            % 选择
            if y_off <= y(i)
                x(i,:) = u;
                y(i) = y_off;
            end
        end
    end
    
    % 找最优解
    [~,idx] = min(y);
    x_best = x(idx,:);
end