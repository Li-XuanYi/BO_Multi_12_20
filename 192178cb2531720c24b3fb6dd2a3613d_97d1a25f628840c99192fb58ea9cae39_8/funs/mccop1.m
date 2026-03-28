classdef mccop1 < PROBLEM
% <multi> <real> <expensive/none>
% m1-充电时间
% m2-最高温度
% m3-SEI膜厚

    properties(Access = private)

    end
    methods
        %% Default settings of the problem
        % 优化问题的一些定义
        function Setting(obj)
            obj.N = 50;
            obj.M = 1;
            obj.maxFE = 1000;
            obj.D = 5;
            obj.lower    = [2,2,2,10,10];
            obj.upper    = [6,5,3,40,30];
            obj.encoding = [1,1,1,2,2];
        end
        %% Calculate objective values
        function PopObj = CalObj(obj,PopDec)
            n = size(PopDec,1);
            PopObj = zeros(n,obj.M);
            % 需要改的地方
            J = [1;0;0];
            m1max = 125; m1min = 45;
            m2max = 1.0; m2min = 7.0;
            m3max = 0.5; m3min = 1.6;
            for i = 1:n
                [m1,m2,m3] = mcc(PopDec(i,4:5),PopDec(i,1:3));
                PopObj(i,1) = J(1)*(m1-m1min)/(m1max-m1min)+...
                    J(2)*(m2-m2min)/(m2max-m2min)+...
                    J(3)*(m3-m3min)/(m3max-m3min);     
            end
        end
    end
end

