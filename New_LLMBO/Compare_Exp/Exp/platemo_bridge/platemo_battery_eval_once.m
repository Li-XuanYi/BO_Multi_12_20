function [PopDec, PopObj, PopCon] = platemo_battery_eval_once(theta, data)
% platemo_battery_eval_once - Evaluate one candidate through Python.

    if ~isfield(data, 'python_module') || isempty(data.python_module)
        data.python_module = 'Compare_Exp.Exp.platemo_eval_helper';
    end
    if ~isfield(data, 'trace_path')
        data.trace_path = '';
    end

    module = py.importlib.import_module(char(data.python_module));
    thetaCell = arrayfun(@(v) py.float(v), double(theta), 'UniformOutput', false);
    payload = module.evaluate_json(py.list(thetaCell), char(data.trace_path));
    decoded = jsondecode(char(payload));

    PopDec = reshape(double(decoded.theta), 1, []);
    PopObj = reshape(double(decoded.objectives), 1, []);
    PopCon = double(decoded.constraint);
end

