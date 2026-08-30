classdef mccop < PROBLEM
% <multi> <real> <expensive/none>
% m1-充电时间
% m2-最高温度
% m3-SEI膜厚
    properties(Access = public)
        init_pop;
        index;
    end

    properties(Access = private)

    end
    methods
        %% Default settings of the problem
        % 优化问题的一些定义
        function Setting(obj)
            obj.N = 20;
            obj.M = 3;
            obj.maxFE = 200;
            obj.D = 5;
            obj.lower    = [2,2,2,10,10];
            obj.upper    = [6,5,3,40,30];
            obj.encoding = [1,1,1,1,1];
        end
        %% Calculate objective values
        function PopObj = CalObj(obj,PopDec)
            n = size(PopDec,1);
            PopObj = zeros(n,obj.M);
            for i = 1:n
                [m1,m2,m3] = mcc(PopDec(i,4:5),PopDec(i,1:3),0.7);
                PopObj(i,1) = m1;
                PopObj(i,2) = m2;
                PopObj(i,3) = m3;        
            end
        end
    end
end

