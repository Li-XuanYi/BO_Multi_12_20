function DB = ParEGO_Source(soh)
    % Parameter Setting
    % 维度5; 迭代次数100; 目标数3
    dim = 5;
    max_iter = 950;
    obj_num = 3;
    up = ones(1,5);
    dn = zeros(1,5);
    [n_w,W] = EqualWeight(30,3);
    W = W + 0.000001;
    % Initialize Dataset
    DB.x = (up - dn).*rand(50,dim) + dn;
    DB.y = Evaluate_Bat_Source(DB.x,soh);
    for iter = 1:max_iter
%        Print HV values
%        y_cal_hv = (DB.y - [700,290,2.5e-10])/([3049.14985470322,313.441659431067,4.27330429607358e-10] - [700,290,2.5e-10]);
%        HV(iter) = Hypervolume(y_cal_hv,[1.1,1.1,1.1]);
%        fprintf([num2str(HV(iter)),' \n'])
        fprintf([num2str(iter+20),' \n'])
        
        % Build GP Surrogate
        idx_w_sel = ceil(n_w*rand());
        w_sel = W(idx_w_sel,:);
        w_sel = 1./w_sel;
        y_norm = (DB.y - min(DB.y))./(max(DB.y) - min(DB.y));
        y_agg = max(( w_sel.* y_norm)')' + 0.05*sum((w_sel .* y_norm),2);
        y_agg_norm = (y_agg - mean(y_agg))./std(y_agg);
        model = fitrgp(DB.x,y_agg_norm);
              
        % Optimize LCB
        y_best = min(y_agg_norm);
        x_new = DE(model,up,dn);
        x_new = (x_new >= up).*up + (x_new < up).*x_new;
        x_new = (x_new <= dn).*dn + (x_new > dn).*x_new;
        % 进行模型仿真，调用接口
        y_new = Evaluate_Bat_Source(x_new,soh);
        
        % Update Dataset
        DB.x = [DB.x;x_new];
        DB.y = [DB.y;y_new];
    end
    % Build population distribution
    [FrontValue,~] = NonDominateSort(DB.y,1);     
    Next = find(FrontValue==1);
    PF_ = DB.y(Next,:);
    PS_ = DB.x(Next,:);
    filename = ['DataSet_Source_ParEGO_1000_' num2str(soh * 100) '.mat'];
    save(filename,'DB')
end

% Differential evaluation for optimizing acquisition function
function x_best = DE(dmodel,up,dn)
[~,d] = size(dn);
x = (up - dn).*rand([30,d]) + dn;
for i = 1:30
    y(i,:) = obj(x(i,:),dmodel);
end
for Iter = 1:200
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
        y_off(i,:) = obj(u(i,:),dmodel);
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
    %sortrows：由小到大以行的方式进行排序，返回排序结果和检索到的数据(按相关度排序)在原始数据中的索引
    
    %开始迭代判断每个个体的前沿面,采用改进的deductive sort，Deb非支配排序算法
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

% Construction of LCB
function y = obj(x,dmodel)
[y_pred,mse] = predict(dmodel,x);
y = y_pred - 0.5*mse;
end