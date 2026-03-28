function [m1,m2,m3,out] = mcc(SOC,I,soh)
%% 多阶段充电协议仿真
% 参数解析
SOC1 = SOC(1)/100; SOC2 = SOC(2)/100; SOC3 = 0.8 - SOC1 - SOC2;
I1 = I(1); I2 = I(2); I3 = I(3);
% 仿真
py_result = py.utils_fun.mcc(SOC1,SOC2,SOC3,I1,I2,I3,soh);
% 结果计算
m1 = double(py_result{1});
m2 = double(py_result{2});
m3 = double(py_result{3});
% 曲线保存
out.U = double(py_result{4});
out.T = double(py_result{5});
out.soc = double(py_result{6});
out.I = double(py_result{7});
end

