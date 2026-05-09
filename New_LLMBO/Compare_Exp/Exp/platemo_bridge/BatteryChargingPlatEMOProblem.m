classdef BatteryChargingPlatEMOProblem < PROBLEM
% BatteryChargingPlatEMOProblem - PlatEMO problem wrapper for New_LLMBO.
%
% The expensive evaluation is delegated to Python via
% Compare_Exp.Exp.platemo_eval_helper so PlatEMO algorithms use the same
% PyBaMM simulator and objective conventions as the Python baselines.

    properties(SetAccess = protected)
        bridgeData = struct();
    end

    methods
        function Setting(obj)
            defaultData = struct( ...
                'project_root', '', ...
                'trace_path', '', ...
                'python_module', 'Compare_Exp.Exp.platemo_eval_helper');
            obj.bridgeData = obj.ParameterSet(defaultData);

            obj.M        = 3;
            obj.D        = 5;
            obj.lower    = [2.0, 2.0, 2.0, 0.10, 0.10];
            obj.upper    = [6.0, 5.0, 3.0, 0.40, 0.30];
            obj.encoding = ones(1, obj.D);
        end

        function Population = Evaluation(obj, varargin)
            PopDec = obj.CalDec(varargin{1});
            N      = size(PopDec, 1);
            PopObj = zeros(N, obj.M);
            PopCon = zeros(N, 1);

            for i = 1:N
                [fixedDec, objValue, conValue] = platemo_battery_eval_once(PopDec(i, :), obj.bridgeData);
                PopDec(i, :) = fixedDec;
                PopObj(i, :) = objValue;
                PopCon(i, 1) = conValue;
            end

            Population = SOLUTION(PopDec, PopObj, PopCon, varargin{2:end});
            obj.FE     = obj.FE + length(Population);
        end

        function R = GetOptimum(obj, ~)
            R = [7200.0, 40.0, 5.0];
        end

        function R = GetPF(obj)
            R = [];
        end
    end
end

