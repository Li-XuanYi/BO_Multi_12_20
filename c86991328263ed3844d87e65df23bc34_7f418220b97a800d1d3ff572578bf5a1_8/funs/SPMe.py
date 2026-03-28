import pybamm
import numpy as np
from gymnasium import spaces
import os


# soc 使用
def cal_soc(c):
    return (c - 872.9651389896292) / (30171.311359086325 - 872.9651389896292)


class SPM:
    def __init__(self, init_v=2.8, init_t=298.15, SOH=1.0):
        # 传递参数
        self.reward = None
        self.sett = {'sample_time': 60,
                     'periodic_test': 20,
                     'number_of_training_episodes': 1000,
                     'number_of_training': 3,
                     'episodes_number_test': 10,
                     'constraints temperature max': 273.15 + 25 + 10,
                     'ambient temperature': 273.15 + 25,  # 严谨
                     'constraints voltage max': 4.2,
                     'max current input': 5}
        # 设置一下日志信息
        # pybamm.set_logging_level("DEBUG")
        # 模型初始化
        options = {"thermal": "lumped"}
        model = pybamm.lithium_ion.SPMe(options=options)
        param = pybamm.ParameterValues("Chen2020")
        param.update(
            {"Current function [A]": "[input]",
             "Upper voltage cut-off [V]": 4.4,
             # 辨识好的电模型参数
             'Negative particle radius [m]': 4.69e-06,
             'Negative electrode active material volume fraction': 0.73,
             'Negative electrode conductivity [S.m-1]': 258.00,
             'Negative electrode diffusivity [m2.s-1]': 3.96e-14,
             'Positive particle radius [m]': 4.17e-06,
             'Positive electrode active material volume fraction': 0.66,
             'Positive electrode conductivity [S.m-1]': 0.22,
             'Positive electrode diffusivity [m2.s-1]': 4.80e-15,
             # 热模型参数
             'Total heat transfer coefficient [W.m-2.K-1]': 17.36,
             'Separator specific heat capacity [J.kg-1.K-1]': 2905.50,
             'Negative electrode specific heat capacity [J.kg-1.K-1]': 2400.56,
             "Positive electrode specific heat capacity [J.kg-1.K-1]": 2715.82,
             'Negative current collector specific heat capacity [J.kg-1.K-1]': 1138.79,
             'Positive current collector specific heat capacity [J.kg-1.K-1]': 1252.81,
             }
        )
        param.update({'Maximum concentration in negative electrode [mol.m-3]': SOH*(33133-1308)+1308})
        # param.update({'Initial inner SEI thickness [m]': 2.5e-09 + 2.5e-09 * (1-SOH) * 1000})
        param.update({'Nominal cell capacity [A.h]': 5.0*SOH})
        # 根据所给的电压初始化参数
        param.set_initial_stoichiometries("{} V".format(init_v))
        self.model = model
        self.param = param
        self.temp = init_t
        self.voltage = init_v
        self.soc = cal_soc(param["Initial concentration in negative electrode [mol.m-3]"])
        self.soc_d = None
        self.temp_d = None
        self.voltage_d = None
        self.sol = None
        self.info = None
        self.done = False
        self.action_space = spaces.Box(low=2, high=6, shape=(1,))
        # 观测空间只提供接口 判断大小 上下限无意义
        self.observation_space = spaces.Box(low=0, high=400, shape=(3,))
        self.spec = self.observation_space
        self.reward_space = np.array([0.0, 0.0])
        # 按 2.5A 计算 120 min
        self._max_episode_steps = 120
        self.step_cnt = 0

    def step(self, action, st=None):
        # 进来的电流在 0-1 判断类型 转成数值 并非如此
        # if isinstance(action, torch.Tensor):
        #     action = action.numpy().item()
        # action = self.actionDeNormalize(action).item()
        # 限制动作的范围
        action = np.clip(action, -2, 2).item()
        action += 4
        # 连续求解状态替换
        if self.sol is not None:
            self.model.set_initial_conditions_from(self.sol)
        # 仿真设置
        simulation = pybamm.Simulation(self.model, parameter_values=self.param)
        # 时间间隔设置 分钟换成秒 提高 pareto 密度
        if st is not None:
            t_eval = np.linspace(0, st, st + 1)
        else:
            t_eval = np.linspace(0, self.sett['sample_time'], 2)
        sol = simulation.solve(t_eval, inputs={"Current function [A]": -action})
        self.voltage = sol["Voltage [V]"].entries[-1]
        self.temp = sol["X-averaged cell temperature [K]"].entries[-1]
        c = sol["R-averaged negative particle concentration [mol.m-3]"].entries[-1][-1]
        self.soc = cal_soc(c)
        self.info = sol.termination
        # 数据的更新
        self.voltage_d = sol["Voltage [V]"].entries
        self.temp_d = sol["X-averaged cell temperature [K]"].entries
        self.soc_d = cal_soc(sol["R-averaged negative particle concentration [mol.m-3]"].entries[-1])
        self.sol = sol
        self.step_cnt += 1
        # 状态打印
        # print(f"step:{self.step_cnt}, 输入电流:{action:.1f}, 电压:{self.voltage:.2f}, 温度:{self.temp:.2f}, SOC:{self.soc:.2f}")
        # 惩罚函数的设置 尽量保证尺度一致 一步在0-1比较好
        # 1. 时间惩罚
        r_step = -self.sol.all_ts[0][-1] / self.sett['sample_time'] * (0.81 - self.soc)
        # 2. 约束惩罚 温度尽可能低 电压别超限
        if self.temp > self.sett['ambient temperature']:
            r_temp = - abs(self.temp - self.sett['ambient temperature']) / 30  # 0-1
        else:
            r_temp = 0
        if self.voltage > self.sett['constraints voltage max']:
            r_volt = -100 * abs(self.voltage - self.sett['constraints voltage max'])
        else:
            r_volt = 0
        r_safe = r_temp + r_volt
        # 3. 老化惩罚
        aval_cap = self.calCap(40, self.temp, action)
        r_aging = - (action / 6) * (80 / aval_cap)  # 最大电流为6，最大可用容量为88
        # 奖励的正数化处理
        r_step = 1 + r_step
        r_safe = max(1 + r_safe, 0)  # 做截断
        r_aging = 1 + r_aging
        # 奖励总和
        self.reward = r_step + r_safe + r_aging
        # 信息打印
        # print(f"step:{self.step_cnt}, 奖励信息, 时间:{r_step:.2f}, 安全:{r_safe:.2f}=<温度:{r_temp:.2f} + 电压:{r_volt:.2f}>, 老化:{r_aging:.2f}")

        # 返回归一化后的状态
        state = np.array([self.voltage, self.temp, self.soc])
        state = self.stateNormalize(state)
        reward = self.reward
        done = self.done

        # 检查充电是否完成
        if self.soc > 0.8:
            done = True
            # self.reset()

        return state, reward, done, {'obj': np.array([r_step, r_safe, r_aging])}, [self.voltage_d, self.temp_d, self.soc_d]

    def reset(self, init_v=2.8, init_t=298.15):
        self.__init__(init_v, init_t)
        # 返回归一化后的状态
        state = np.array([self.voltage, self.temp, self.soc])
        self.stateNormalize(state)
        return state

    # 用参数表示比较好
    def stateNormalize(self, state):
        state[0] = (state[0] - 2.7) / (4.7 - 2.7)
        state[1] = (state[1] - 290) / (320 - 290)
        return state

    def stateDeNormalize(self, state):
        state[0] = state[0] * 2 + 2.7
        state[1] = state[1] * 30 + 290
        return state

    def actionNormalize(self, action):
        return (action - self.action_space.low) * 2 / (self.action_space.high - self.action_space.low) - 1

    def actionDeNormalize(self, action):
        return (action + 1) / 2 * (self.action_space.high - self.action_space.low) + self.action_space.low

    # 返回该状态下的可用容量
    def calCap(self, soc, temp, current):
        tmp = (2896.6 * soc + 7411.2) * np.exp((-31500 + 152.5 * current) / (8.314 * temp))
        cap = (20 / tmp) ** (1 / 0.57)
        return cap

    def seed(self, num):
        pass

    def close(self):
        pass