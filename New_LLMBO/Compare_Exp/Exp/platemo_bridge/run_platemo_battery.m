function run_platemo_battery(configPath)
% run_platemo_battery - Batch entrypoint used by platemo_runner.py.
%
% The config file is JSON to avoid brittle command-line quoting.

    cfg = jsondecode(fileread(configPath));

    projectRoot = char(cfg.project_root);
    platemoRoot = char(cfg.platemo_root);
    outputDir   = char(cfg.output_dir);
    tracePath   = char(cfg.trace_path);

    if isfield(cfg, 'python_executable') && ~isempty(cfg.python_executable)
        try
            pyenv('Version', char(cfg.python_executable));
        catch ME
            warning('Failed to set MATLAB Python interpreter: %s', ME.message);
        end
    end

    if int64(py.sys.path.count(projectRoot)) == 0
        py.sys.path.insert(int32(0), projectRoot);
    end

    addpath(genpath(platemoRoot));
    addpath(fileparts(mfilename('fullpath')));

    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end
    if exist(tracePath, 'file')
        delete(tracePath);
    end

    rng(double(cfg.seed), 'twister');

    algName = upper(char(cfg.algorithm));
    switch algName
        case 'DISK'
            algFcn = @DISK;
            if isfield(cfg, 'algorithm_parameters')
                algSpec = [{algFcn}, num2cell(double(cfg.algorithm_parameters(:)'))];
            else
                algSpec = {algFcn, 60, 5};
            end
        case 'PIMD'
            algFcn = @PIMD;
            if isfield(cfg, 'algorithm_parameters')
                algSpec = [{algFcn}, num2cell(double(cfg.algorithm_parameters(:)'))];
            else
                algSpec = {algFcn, 15, 5};
            end
        otherwise
            error('Unsupported PlatEMO algorithm: %s', algName);
    end

    bridgeData = struct( ...
        'project_root', projectRoot, ...
        'trace_path', tracePath, ...
        'python_module', 'Compare_Exp.Exp.platemo_eval_helper');

    [decs, objs, cons] = platemo( ...
        'algorithm', algSpec, ...
        'problem', {@BatteryChargingPlatEMOProblem, bridgeData}, ...
        'N', double(cfg.population_size), ...
        'maxFE', double(cfg.n_evals), ...
        'save', 0, ...
        'run', double(cfg.seed));

    result = struct();
    result.algorithm = algName;
    result.seed = double(cfg.seed);
    result.n_evals = double(cfg.n_evals);
    result.population_size = double(cfg.population_size);
    result.final_decs = decs;
    result.final_objs = objs;
    result.final_cons = cons;

    fid = fopen(char(cfg.final_population_path), 'w');
    cleanup = onCleanup(@() fclose(fid));
    fwrite(fid, jsonencode(result), 'char');
end
