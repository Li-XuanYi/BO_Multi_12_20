import time
from SPMe import SPM
import numpy as np
import logging

# 配置日志
logging.basicConfig(
    filename='mcc.log',  # 日志文件名
    level=logging.DEBUG,  # 设置日志级别为 DEBUG
    format='%(asctime)s %(levelname)s: %(message)s',  # 日志格式
    # encoding='utf-8'
)
"""
输入：soc1,soc2,soc3,i1,i2,i3
     soc范围在0-1，i范围在2-6A
输出：char_time,temp_rise,capa_aging
OKane2022/Mohtat2020/ORegan2022/Ecker2015
"""


def mcc(soc1, soc2, soc3, i1, i2, i3, SOH, model_type="ours"):
    start_time = time.time()
    logging.info(f"充电配置为三阶段恒流充电协议，soc分为{soc1:.2f}, {soc2:.2f}, {soc3:.2f},电流分为{i1:.2f}, {i2:.2f}, {i3:.2f}...")
    print(f"充电配置为三阶段恒流充电协议，soc分为{soc1:.2f}, {soc2:.2f}, {soc3:.2f},电流分为{i1:.2f}, {i2:.2f}, {i3:.2f}...")
    model = SPM(SOH=SOH, model_type=model_type)
    soc_list = [soc1, soc2, soc3]
    bat_cap = model.param["Nominal cell capacity [A.h]"]
    # Ecker2015模型需要将电流放大3倍（不然电流太小）
    CURRENT_SCALE_FACTOR_Ecker2015 = 3  # Ecker2015模型电流缩放因子
    if model_type == "Ecker2015":
        i1 = i1 * CURRENT_SCALE_FACTOR_Ecker2015
        i2 = i2 * CURRENT_SCALE_FACTOR_Ecker2015
        i3 = i3 * CURRENT_SCALE_FACTOR_Ecker2015
        logging.info(f"Ecker2015模型电流已缩放{CURRENT_SCALE_FACTOR_Ecker2015}倍，缩放后电流={[i1, i2, i3]}")
    input_current = [i * bat_cap / 5 for i in [i1, i2, i3]]
    steps = [bat_cap*SOH / i * 60 * soc for i, soc in zip(input_current, soc_list)]
    logging.info(f"计算得到各阶段的充电时间为, 阶段1: {steps[0]:.2f} 分钟, 阶段2: {steps[1]:.2f} 分钟, 阶段3: {steps[2]:.2f} 分钟...")
    voltage = [model.voltage]
    temperature = [model.temp]
    soc = [model.soc]
    current = [0]
    for i in range(len(steps)):
        # 改用精确的秒数
        info1, info2, done, info4, state_vec = model.step(input_current[i], round(60 * steps[i]))
        voltage.extend(state_vec[0][1:])
        temperature.extend(state_vec[1][1:])
        soc.extend(state_vec[2][1:])
        current.extend(np.full(round(steps[i] * 60), input_current[i]))
    char_time = len(soc) - 1            # 单位：分钟
    temp_rise = max(temperature) - 298.15
    # logging.info(np.mean(soc) * 100, np.mean(temperature), np.mean(current))
    capa_aging = model.calCap(np.mean(soc) * 100, np.mean(temperature), np.mean(current))
    # print(capa_aging)
    CAP = model.param["Nominal cell capacity [A.h]"]
    logging.info(f"电池容量为{CAP}Ah")
    logging.info(f"最终指标为, 充电时间: {char_time} 分钟, 温升: {temp_rise:.2f} 度, 消耗寿命: {5/capa_aging*100:.2f} %...")
    print(f"最终指标为, 充电时间: {char_time} 分钟, 温升: {temp_rise:.2f} 度, 消耗寿命: {5/capa_aging*100:.2f} %...")
    end_time = time.time()
    logging.info(f"程序运行时间：{end_time-start_time:.4f} 秒")
    return char_time, temp_rise, CAP/capa_aging*100, voltage, temperature, soc, current

"OKane2022/Mohtat2020(无温度)/ORegan2022/Ecker2015"
mcc(0.40, 0.30, 0.10, 6, 5, 3, 0.7, "ours")
