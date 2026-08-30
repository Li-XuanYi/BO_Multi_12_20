function [m1,m2,m3,out] = mcc_5stage(soc1,soc2,soc3,soc4,soc5,i1,i2,i3,i4,i5,soh)
    % 直接调用你最开始的Python函数
     py_result = py.utils_fun.mcc_5stage(soc1,soc2,soc3,soc4,soc5,i1,i2,i3,i4,i5, soh);

    m1 = double(py_result{1});
    m2 = double(py_result{2});
    m3 = double(py_result{3});

    out.U  = double(py_result{4});
    out.T  = double(py_result{5});
    out.soc= double(py_result{6});
    out.I  = double(py_result{7});
end