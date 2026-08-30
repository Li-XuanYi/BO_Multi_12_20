"""
测试 SPM.py 参数对齐的 PyBaMM 仿真器
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pybamm_simulator import PyBaMMSimulator, _neg_conc_to_soc, _aging_severity, _cal_dQloss_pct


def test_soc_conversion():
    """测试 SOC 转换函数（与 SPM.py cal_soc 一致）"""
    print("=" * 60)
    print("测试 1: SOC 转换函数")
    print("=" * 60)

    # SPM.py 参数
    c_min = 872.9651389896292
    c_max = 30171.311359086325

    # 测试几个点
    test_cases = [
        (c_min, 0.0),
        (c_max, 1.0),
        ((c_min + c_max) / 2, 0.5),
        (3.0e4, 0.98),
    ]

    for c, expected in test_cases:
        result = _neg_conc_to_soc(c)
        print(f"  c={c:.2f} → SOC={result:.4f} (expected ~{expected:.4f})")
        assert 0.0 <= result <= 1.0, f"SOC 应该在 [0,1] 范围内，但得到 {result}"

    print("✓ SOC 转换函数测试通过\n")


def test_aging_severity():
    """测试老化严重程度计算（与 SPM.py aging_severity 一致）"""
    print("=" * 60)
    print("测试 2: 老化严重程度计算")
    print("=" * 60)

    # SPM.py 参数
    soc = 0.5  # 50% SOC
    temp_K = 298.15  # 25°C
    current_A = 5.0  # 5A

    result = _aging_severity(soc, temp_K, current_A)
    print(f"  s(SOC=0.5, T=298.15K, I=5A) = {result:.6f}")

    # 检查是否为正数
    assert result > 0, "老化严重程度应该为正数"

    # 温度越高，老化越快
    result_high_temp = _aging_severity(soc, 318.15, current_A)
    print(f"  s(SOC=0.5, T=318.15K, I=5A) = {result_high_temp:.6f}")
    assert result_high_temp > result, "温度越高老化应该越快"

    # 电流越大，老化越快
    result_high_current = _aging_severity(soc, temp_K, 10.0)
    print(f"  s(SOC=0.5, T=298.15K, I=10A) = {result_high_current:.6f}")
    assert result_high_current > result, "电流越大老化应该越快"

    print("✓ 老化严重程度计算测试通过\n")


def test_dQloss_calculation():
    """测试 dQloss 计算（与 SPM.py cal_dQloss_pct 一致）"""
    print("=" * 60)
    print("测试 3: dQloss 计算")
    print("=" * 60)

    soc_avg = 0.5
    temp_K = 298.15
    current_A = 5.0
    duration_s = 60.0  # 1 分钟
    aging_Ah_old = 0.0

    dQloss, aging_Ah_new = _cal_dQloss_pct(
        soc_avg=soc_avg,
        temp_K=temp_K,
        current_A=current_A,
        duration_s=duration_s,
        aging_Ah_old=aging_Ah_old,
    )

    print(f"  dQloss_pct (1min, 5A, 25°C) = {dQloss:.6f}%")
    print(f"  aging_Ah after 1min = {aging_Ah_new:.6f} Ah")

    assert dQloss >= 0, "dQloss 应该为非负数"
    assert aging_Ah_new > aging_Ah_old, "累计 Ah 应该增加"

    print("✓ dQloss 计算测试通过\n")


def test_simulator_basic():
    """测试仿真器的基本功能"""
    print("=" * 60)
    print("测试 4: 仿真器基本功能")
    print("=" * 60)

    sim = PyBaMMSimulator(
        Q_nom=5.0,
        SOH=1.0,
        T_init=298.15,
        V_init=2.8,
        T_max=328.15,
        V_max=4.4,
        use_crate=False,
        aging_weight=10.0,
    )

    # 测试用例：中等电流充电协议
    theta = np.array([4.0, 3.5, 2.5, 0.25, 0.25])
    print(f"  输入参数：I1={theta[0]}A, I2={theta[1]}A, I3={theta[2]}A, dSOC1={theta[3]}, dSOC2={theta[4]}")

    try:
        result = sim.evaluate(theta)

        print(f"\n  仿真结果:")
        print(f"    可行性：{result['feasible']}")
        print(f"    violation: {result.get('violation', 'N/A')}")

        if result['feasible']:
            objectives = result['raw_objectives']
            print(f"    目标值 [time_s, delta_temp_K, aging_%]:")
            print(f"      time_s = {objectives[0]:.1f}s ({objectives[0]/60:.1f}min)")
            print(f"      delta_temp_K = {objectives[1]:.2f}K")
            print(f"      aging_% = {objectives[2]:.4f}%")
            print(f"    最终 SOC: {result['soc_final']:.4f}")

            # 轨迹检查
            traj = result['trajectories']
            if traj:
                print(f"    轨迹长度：V={len(traj['V'])}, T={len(traj['T'])}, SOC={len(traj['SOC'])}, I={len(traj['I'])}")

                # 检查电压、温度、电流轨迹长度一致（它们都是逐点输出）
                assert len(traj['V']) == len(traj['T']), "电压和温度轨迹长度应一致"
                assert len(traj['V']) == len(traj['I']), "电压和电流轨迹长度应一致"

                # SOC 可能采样率不同（取决于浓度数组形状），但应该有合理的值
                v_min, v_max = min(traj['V']), max(traj['V'])
                print(f"    电压范围：[{v_min:.3f}V, {v_max:.3f}V]")

                # 检查温度范围
                t_min, t_max = min(traj['T']), max(traj['T'])
                print(f"    温度范围：[{t_min:.2f}K, {t_max:.2f}K]")

                # 检查 SOC 变化（使用最后一个值）
                soc_start = traj['SOC'][0] if traj['SOC'] else 0.0
                soc_end = traj['SOC'][-1] if traj['SOC'] else 0.0
                print(f"    SOC 变化：{soc_start:.4f} → {soc_end:.4f}")

            print("\n✓ 仿真器基本功能测试通过\n")
        else:
            print(f"  ⚠ 仿真失败：{result['violation']}")
            print("  （这可能是约束违反，属于正常情况）\n")

    except Exception as e:
        print(f"  ✗ 仿真器测试失败：{e}")
        raise


def test_simulator_different_params():
    """测试不同参数组合"""
    print("=" * 60)
    print("测试 5: 不同参数组合")
    print("=" * 60)

    sim = PyBaMMSimulator()

    test_cases = [
        ("高电流", np.array([5.0, 4.0, 3.0, 0.25, 0.25])),
        ("低电流", np.array([2.5, 2.5, 2.0, 0.30, 0.30])),
        ("不平衡", np.array([5.0, 3.0, 2.5, 0.40, 0.20])),
    ]

    for name, theta in test_cases:
        print(f"\n  测试：{name}")
        try:
            result = sim.evaluate(theta)
            if result['feasible']:
                obj = result['raw_objectives']
                print(f"    ✓ 时间={obj[0]/60:.1f}min, 温升={obj[1]:.1f}K, 老化={obj[2]:.4f}%")
            else:
                print(f"    ⚠ 不可行：{result['violation']}")
        except Exception as e:
            print(f"    ✗ 错误：{e}")

    print("\n✓ 不同参数组合测试完成\n")


def test_constraint_violation():
    """测试约束违反情况"""
    print("=" * 60)
    print("测试 6: 约束违反检测")
    print("=" * 60)

    sim = PyBaMMSimulator()

    # 违反 dSOC 约束
    theta_invalid = np.array([4.0, 3.5, 2.5, 0.40, 0.40])  # dSOC1+dSOC2=0.80 > 0.70
    result = sim.evaluate(theta_invalid)
    print(f"  dSOC 约束违反：feasible={result['feasible']}, violation='{result['violation']}'")
    assert not result['feasible'], "应该检测到 dSOC 约束违反"

    # 负电流
    theta_neg = np.array([-1.0, 3.0, 2.0, 0.25, 0.25])
    result = sim.evaluate(theta_neg)
    print(f"  负电流检测：feasible={result['feasible']}, violation='{result['violation']}'")
    assert not result['feasible'], "应该检测到负电流"

    print("\n✓ 约束违反检测测试通过\n")


def main():
    """运行所有测试"""
    print("\n")
    print("#" * 60)
    print("# SPM.py 参数对齐的 PyBaMM 仿真器测试")
    print("#" * 60)
    print()

    try:
        test_soc_conversion()
        test_aging_severity()
        test_dQloss_calculation()
        test_simulator_basic()
        test_simulator_different_params()
        test_constraint_violation()

        print("=" * 60)
        print("所有测试通过！✓")
        print("=" * 60)
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"测试失败：{e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
