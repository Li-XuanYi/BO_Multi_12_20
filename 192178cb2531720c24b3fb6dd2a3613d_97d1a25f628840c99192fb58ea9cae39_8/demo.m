% 每个实验跑五次
for i = 1:5
    tic;
    % 上下界设定
    up = ones(1,5);
    dn = zeros(1,5);
    % 初始化种群（随机的）
    init_pop = (up - dn).*rand(20,5) + dn;
    % 两种方法的测试
    [DB_EEI,betalist] = TrOpt(init_pop);
    % DB_ParEGO = ParEGO(init_pop);
    % 存储得到的数据
    EEI_DATA{i} = DB_EEI;
    Betalist{i} = betalist;
    % ParEGO_DATA{i} = DB_ParEGO;
    save 'EEI_DATA' EEI_DATA
    save 'Betalist' Betalist
    % save 'ParEGO_DATA' ParEGO_DATA
    toc;
end
