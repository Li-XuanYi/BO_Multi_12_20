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
输入：soc1,soc2,soc3,soc4,soc5（总和=0.8）,i1,i2,i3,i4,i5
     soc范围在0-1（总和强制为0.8），i范围在2-6A
输出：char_time,temp_rise,capa_aging, voltage, temperature, soc, current
"""


def mcc_5stage(soc1, soc2, soc3, soc4, soc5, i1, i2, i3, i4, i5, SOH):
    # ========== 核心约束：强制5个SOC之和为0.8 ==========
    soc_sum = soc1 + soc2 + soc3 + soc4 + soc5
    if not np.isclose(soc_sum, 0.8):
        # 按比例归一化到总和0.8（避免输入错误导致约束不满足）
        scale_factor = 0.8 / soc_sum
        soc1, soc2, soc3, soc4, soc5 = [soc * scale_factor for soc in [soc1, soc2, soc3, soc4, soc5]]
        logging.warning(f"输入SOC总和({soc_sum:.4f})≠0.8，已按比例归一化："
                        f"soc1={soc1:.4f}, soc2={soc2:.4f}, soc3={soc3:.4f}, soc4={soc4:.4f}, soc5={soc5:.4f}")

    start_time = time.time()
    logging.info(f"充电配置为五阶段恒流充电协议，SOC分为{soc1:.4f}, {soc2:.4f}, {soc3:.4f}, {soc4:.4f}, {soc5:.4f},"
                 f"电流分为{i1:.2f}, {i2:.2f}, {i3:.2f}, {i4:.2f}, {i5:.2f}...")

    model = SPM(SOH=SOH)
    input_current = [i1, i2, i3, i4, i5]  # 五阶段电流
    soc_list = [soc1, soc2, soc3, soc4, soc5]  # 五阶段SOC占比（总和=0.8）

    # ========== 计算五阶段充电时间（沿用原公式，适配5阶段） ==========
    steps = [5 * SOH / i * 60 * soc for i, soc in zip(input_current, soc_list)]
    logging.info(f"计算得到各阶段的充电时间为："
                 f"阶段1: {steps[0]:.2f} 分钟, 阶段2: {steps[1]:.2f} 分钟, 阶段3: {steps[2]:.2f} 分钟,"
                 f"阶段4: {steps[3]:.2f} 分钟, 阶段5: {steps[4]:.2f} 分钟...")

    # 初始化状态变量（保留原逻辑）
    voltage = [model.voltage]
    temperature = [model.temp]
    soc = [model.soc]
    current = [0]

    # ========== 五阶段充电仿真（循环5次） ==========
    for i in range(len(steps)):
        # 改用精确的秒数，沿用原step调用逻辑
        info1, info2, done, info4, state_vec = model.step(input_current[i] - 4, round(60 * steps[i]))
        voltage.extend(state_vec[0][1:])
        temperature.extend(state_vec[1][1:])
        soc.extend(state_vec[2][1:])
        current.extend(np.full(round(steps[i] * 60), input_current[i]))

    # ========== 计算输出指标（与原逻辑一致） ==========
    char_time = len(soc) - 1  # 单位：分钟
    temp_rise = max(temperature) - 298.15
    capa_aging = model.calCap(np.mean(soc) * 100, np.mean(temperature), np.mean(current))
    CAP = model.param["Nominal cell capacity [A.h]"]

    # 日志与打印输出（适配五阶段）
    logging.info(f"电池容量为{CAP}Ah")
    logging.info(
        f"最终指标为, 充电时间: {char_time} 分钟, 温升: {temp_rise:.2f} 度, 消耗寿命: {5 / capa_aging * 100:.2f} %...")
    print(
        f"最终指标为, 充电时间: {char_time} 分钟, 温升: {temp_rise:.2f} 度, 消耗寿命: {5 / capa_aging * 100:.2f} %...")

    end_time = time.time()
    logging.info(f"程序运行时间：{end_time - start_time:.4f} 秒")

    return char_time, temp_rise, 5 / capa_aging * 100, voltage, temperature, soc, current


# ========== 测试调用（示例：5个SOC之和=0.8） ==========
# 示例输入：SOC=[0.2,0.2,0.15,0.15,0.1]（总和=0.8），电流=[6,5,4,3,2]，SOH=1
mcc_5stage(0.2, 0.2, 0.15, 0.15, 0.1, 6, 6, 5, 5, 3, 0.7)