function y = Evaluate_Bat_Source(x,soh)
    up = [6,5,3,40,30];
    dn = [2,2,2,10,10];
    x = x.*(up - dn) + dn;
    [m,~] = size(x);
    y = zeros(m,3);
    for i = 1:m
        % 调用 Python 模型
        [m1,m2,m3,~] = mcc(x(i,4:5),x(i,1:3),soh);
        y(i,1) = m1;
        y(i,2) = m2;
        y(i,3) = m3;
    end
end