from SPMe import SPM
from utils_fun import mcc


def add(var1, var2, var3, var4, var5, var6):
    metric1, metric2, metric3 = mcc(var1, var2, var3, var4, var5, var6)
    return metric1, metric2, metric3

