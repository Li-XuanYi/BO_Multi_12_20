"""
多输出高斯过程模型模块（FrameWork.md §3 实现）

约束 C-2 实现：
- W^(t) 仅由λ权重组合：W^(t) = (w0*W_time + w1*W_temp + w2*W_aging) / sum(w)
- 不使用 W_data 融合

决策空间：[I1, SOC1, I2]，d=3
目标空间：[time, temp, aging]，m=3
"""

import numpy as np
from typing import List, Dict, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
import warnings

# 导入自定义模块
from models.kernels import CompositeKernel, CouplingKernel, ensure_psd
from models.gradients import GPGradientComputer
# 导入 config
from config import get_algorithm_param

warnings.filterwarnings('ignore', category=UserWarning)


class MOGPModel:
    """
    多目标高斯过程模型（3D 版本 - FrameWork.md §3）

    决策空间：[I1, SOC1, I2]，d=3
    目标空间：[time, temp, aging]，m=3

    约束 C-2 实现：
    - W^(t) 仅由λ权重组合：W^(t) = (w0*W_time + w1*W_temp + w2*W_aging) / sum(w)
    - 不使用 W_data 融合
    """

    def __init__(
        self,
        use_coupling: bool = True,
        gamma_init: float = 0.1,  # 约束 C-6
        n_dims: int = 3,          # 3D 决策空间
        verbose: bool = False
    ):
        """
        初始化多目标 GP 模型

        参数：
            use_coupling: 是否使用物理耦合核
            gamma_init: 初始耦合强度（约束 C-6: 0.1）
            n_dims: 决策空间维度（默认 3 - FrameWork.md §0）
            verbose: 详细输出
        """
        self.use_coupling = use_coupling
        self.gamma = gamma_init
        self.n_dims = n_dims
        self.param_keys = ['I1', 'SOC1', 'I2']  # 3D 决策变量
        self.ard_length_scales = None  # LLM 驱动的长度尺度
        self.verbose = verbose

        # GP 梯度计算器
        self.gradient_computer = GPGradientComputer(
            epsilon=1e-4,
            verbose=verbose
        )

        # 存储训练好的 GP 列表
        self.gp_list = None
        self.coupling_matrix = None
        self.W_llm = None  # LLM 推理的耦合矩阵（仅在 Touchpoint 1a 使用）

    def _extract_training_data(self, database: List[Dict]):
        """
        从数据库提取训练数据

        参数：
            database: legacy 格式评估历史

        返回：
            X_train: (N, 3) 参数矩阵
            y_dict: {'time': array, 'temp': array, 'aging': array}
            valid_data: 有效数据列表
        """
        valid_data = [r for r in database if r['valid']]

        if len(valid_data) < 3:
            raise ValueError(f"有效数据不足：{len(valid_data)} < 3")

        X_train = np.array([[
            r['params'][key] for key in self.param_keys
        ] for r in valid_data])

        y_dict = {
            'time': np.array([r['time'] for r in valid_data]),
            'temp': np.array([r['temp'] for r in valid_data]),
            'aging': np.array([r['aging'] for r in valid_data])
        }

        return X_train, y_dict, valid_data

    def train_pilot_gp(self, database: List[Dict]) -> List[GaussianProcessRegressor]:
        """
        训练 Pilot GP（用于推理耦合矩阵）

        特点：标准 ARD Matern 核，提供干净的梯度信息

        参数：
            database: 评估历史（已经过 DataTransformer 变换）

        返回：
            gp_list: [gp_time, gp_temp, gp_aging]
        """
        if self.verbose:
            print(f"\n  [训练 Pilot GP]")

        X_train, y_dict, valid_data = self._extract_training_data(database)

        if self.verbose:
            print(f"    训练数据：{len(valid_data)}个有效点")
            print(f"    Time 范围：[{np.min(y_dict['time']):.1f}, {np.max(y_dict['time']):.1f}]")
            print(f"    Temp 范围：[{np.min(y_dict['temp']):.1f}, {np.max(y_dict['temp']):.1f}]")
            print(f"    Aging 范围：[{np.min(y_dict['aging']):.4f}, {np.max(y_dict['aging']):.4f}]")

        # ARD Matern 核（length_scale=np.ones(d) 启用各向异性）
        kernel = C(1.0, constant_value_bounds=(1e-3, 1e3)) * \
                 Matern(nu=2.5,
                        length_scale=np.ones(self.n_dims),
                        length_scale_bounds=(1e-2, 1e2))

        gp_list = []
        for obj_name, y_key in [('Time', 'time'), ('Temp', 'temp'), ('Aging', 'aging')]:
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=get_algorithm_param('gp', 'n_restarts_optimizer', 5),
                alpha=get_algorithm_param('gp', 'alpha', 1e-6),
                normalize_y=get_algorithm_param('gp', 'normalize_y', True),
                random_state=42
            )
            gp.fit(X_train, y_dict[y_key])
            gp_list.append(gp)

            if self.verbose:
                print(f"    GP-{obj_name}: 训练完成")

        return gp_list

    def train_main_gp(
        self,
        database: List[Dict],
        coupling_matrix: Optional[np.ndarray] = None
    ) -> List[GaussianProcessRegressor]:
        """
        训练 Main GP（用于采集优化）

        特点：复合核 = ARD Matern + γ·物理耦合核，支持 LLM 长度尺度

        参数：
            database: 评估历史（已经过 DataTransformer 变换）
            coupling_matrix: 耦合矩阵 (3x3)

        返回：
            gp_list: [gp_time, gp_temp, gp_aging]
        """
        if self.verbose:
            print(f"\n  [训练 Main GP]")

        X_train, y_dict, valid_data = self._extract_training_data(database)

        if self.verbose:
            print(f"    训练数据：{len(valid_data)}个有效点")

        # 选择核函数
        if self.use_coupling and coupling_matrix is not None:
            kernel = CompositeKernel(
                coupling_matrix=coupling_matrix,
                gamma=self.gamma,
                n_dims=self.n_dims
            )

            if self.ard_length_scales is not None:
                kernel.set_ard_length_scales(self.ard_length_scales)
                if self.verbose:
                    print(f"    已设置 ARD 长度尺度：{self.ard_length_scales}")

            if self.verbose:
                print(f"    使用耦合核 (γ={self.gamma:.3f})")
        else:
            kernel = C(1.0, constant_value_bounds=(1e-3, 1e3)) * \
                     Matern(nu=2.5,
                            length_scale=np.ones(self.n_dims),
                            length_scale_bounds=(1e-2, 1e2))
            if self.verbose:
                print(f"    使用标准 ARD Matern 核")

        gp_list = []
        for obj_name, y_key in [('Time', 'time'), ('Temp', 'temp'), ('Aging', 'aging')]:
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=get_algorithm_param('gp', 'n_restarts_optimizer', 5),
                alpha=get_algorithm_param('gp', 'alpha', 1e-6),
                normalize_y=get_algorithm_param('gp', 'normalize_y', True),
                random_state=42
            )
            gp.fit(X_train, y_dict[y_key])
            gp_list.append(gp)

            if self.verbose:
                print(f"    GP-{obj_name}: 训练完成")

        self.gp_list = gp_list
        self.coupling_matrix = coupling_matrix
        return gp_list

    def estimate_coupling_from_gradients(
        self,
        gp_list: List[GaussianProcessRegressor],
        database: List[Dict],
        n_samples: int = 10
    ) -> np.ndarray:
        """
        从 GP 梯度估计耦合矩阵 W_data (3×3)

        方法：梯度外积平均 → 归一化 → PSD 保证
        """
        if self.verbose:
            print(f"\n  [估计数据驱动耦合矩阵]")

        X_train, y_dict, valid_data = self._extract_training_data(database)

        if len(valid_data) < n_samples:
            n_samples = len(valid_data)

        indices = np.random.choice(len(valid_data), size=n_samples, replace=False)
        X_samples = X_train[indices]

        if self.verbose:
            print(f"    采样点数：{n_samples}")

        W_data = self.gradient_computer.estimate_coupling_matrix(
            gp_list=gp_list,
            X_samples=X_samples,
            method='outer_product'
        )

        if self.verbose:
            print(f"    耦合矩阵已估计")

        return W_data

    def build_W_t(
        self,
        weights: np.ndarray,
        W_time: np.ndarray,
        W_temp: np.ndarray,
        W_aging: np.ndarray
    ) -> np.ndarray:
        """
        构建 W^(t) 耦合矩阵 - FrameWork.md Eq.11'

        约束 C-2 实现：
        - W^(t) 仅由λ权重组合，不存在 W_data 融合
        - 没有融合系数 alpha
        - 权重就是当轮 Tchebycheff 权重λ

        公式：W^(t) = (w[0]*W_time + w[1]*W_temp + w[2]*W_aging) / (w[0]+w[1]+w[2])

        注意：
        - W_time, W_temp, W_aging 由 LLM 在 Touchpoint 1a 推断
        - 每轮迭代根据当前 Tchebycheff 权重λ组合

        参数：
            weights: Tchebycheff 权重λ (3,)
            W_time: LLM 推断的时间耦合矩阵 (3×3)
            W_temp: LLM 推断的温度耦合矩阵 (3×3)
            W_aging: LLM 推断的老化耦合矩阵 (3×3)

        返回：
            W_t: 组合后的耦合矩阵 (3×3)
        """
        assert len(weights) == 3, f"权重向量必须是 3 维，得到{len(weights)}"
        assert W_time.shape == (3, 3), f"W_time 必须是 3×3，得到{W_time.shape}"
        assert W_temp.shape == (3, 3), f"W_temp 必须是 3×3，得到{W_temp.shape}"
        assert W_aging.shape == (3, 3), f"W_aging 必须是 3×3，得到{W_aging.shape}"

        if self.verbose:
            print(f"  [构建 W^(t)] λ= ({weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f})")

        # 约束 C-2 公式：W^(t) = (w0*W_time + w1*W_temp + w2*W_aging) / sum(w)
        w_sum = weights.sum()
        W_t = (weights[0] * W_time + weights[1] * W_temp + weights[2] * W_aging) / w_sum

        # 对称化
        W_t = (W_t + W_t.T) / 2.0

        # 强制对角线为 1
        np.fill_diagonal(W_t, 1.0)

        # PSD 保证
        eps_psd = get_algorithm_param('composite_kernel', 'eps_psd', 1e-6)
        W_t = ensure_psd(W_t, eps=eps_psd)

        # 再次强制对角线为 1
        np.fill_diagonal(W_t, 1.0)

        return W_t

    def fit(self, database: List[Dict]):
        """
        完整的两阶段训练流程（便捷接口）
        """
        gp_pilot = self.train_pilot_gp(database)
        W_data = self.estimate_coupling_from_gradients(gp_pilot, database, n_samples=10)
        self.train_main_gp(database, coupling_matrix=W_data)

    def set_llm_coupling_matrix(self, W_llm: np.ndarray):
        """设置 LLM 推理的耦合矩阵 (3×3)"""
        self.W_llm = W_llm
        if self.verbose:
            print(f"  [MOGPModel] LLM 耦合矩阵已设置")

    def set_ard_length_scales(self, length_scales):
        """
        设置 ARD 长度尺度（LLM 驱动）

        参数：
            length_scales: np.ndarray, 形状 (d,)
        """
        length_scales = np.asarray(length_scales, dtype=float)
        assert len(length_scales) == self.n_dims, \
            f"Length scales 维度不匹配：期望{self.n_dims}, 得到{len(length_scales)}"
        assert np.all(length_scales > 0), "All length scales must be > 0"

        self.ard_length_scales = length_scales.copy()

        if self.verbose:
            print(f"  [MOGPModel] ARD 长度尺度已设置：{self.ard_length_scales}")

    def update_coupling_matrix(self, method: str = 'gradient'):
        """更新耦合矩阵（占位，由 main.py 外部调用实现）"""
        if not self.use_coupling or self.gp_list is None:
            return
        if self.verbose:
            print(f"  [更新耦合矩阵] method={method}")

    def update_gamma(self, improvement_rate: float):
        """更新 gamma 参数（由 main.py 的 f_min 自适应外部处理）"""
        pass

    # ── main.py 兼容桥接方法 ─────────────────────────────────────

    def set_coupling_matrix(self, W: np.ndarray):
        """设置当前耦合矩阵（main.py 的 W^(t) 构建后调用）"""
        self.coupling_matrix = np.asarray(W, dtype=float)

    def set_gamma(self, gamma: float):
        """设置当前 gamma 值"""
        self.gamma = float(gamma)

    def train(self, database: List[Dict], weights: np.ndarray = None):
        """
        训练 Main GP — main.py 统一入口。

        等价于 train_main_gp(database, self.coupling_matrix)。
        weights 参数保留接口兼容，不参与实际训练（W^(t) 已提前由 set_coupling_matrix 设置）。
        """
        return self.train_main_gp(database, coupling_matrix=self.coupling_matrix)

    def get_gp_list(self) -> List[GaussianProcessRegressor]:
        """返回当前 GP 列表 [gp_time, gp_temp, gp_aging]"""
        return self.gp_list if self.gp_list is not None else []


# ============================================================
# 快速测试（3D 版本）
# ============================================================
if __name__ == "__main__":
    print("测试 MOGPModel（3D 参数空间 - FrameWork.md §0）...")

    # ===== 创建 3D 虚拟数据库 =====
    np.random.seed(42)
    fake_database = []

    for i in range(20):
        fake_database.append({
            'params': {
                'I1': np.random.uniform(3.0, 7.99),
                'SOC1': np.random.uniform(0.1, 0.7),
                'I2': np.random.uniform(2.0, 7.99)
            },
            'time': np.random.uniform(600, 3600),   # 秒
            'temp': np.random.uniform(300, 315),     # K
            'aging': np.random.uniform(-5.0, -1.5),  # log10 空间
            'valid': True
        })

    # 初始化模型
    model = MOGPModel(use_coupling=True, gamma_init=0.1, n_dims=3, verbose=True)

    # ===== 测试 1: Pilot GP 训练 =====
    print("\n" + "="*60)
    print("测试 1: Pilot GP 训练（3D ARD Matern）")
    print("="*60)

    gp_list_pilot = model.train_pilot_gp(fake_database)
    print(f"\nOK Pilot GP 训练成功，得到{len(gp_list_pilot)}个模型")

    # 验证预测
    x_test = np.array([[4.0, 0.4, 3.0]])
    for i, name in enumerate(['Time', 'Temp', 'Aging']):
        mu, std = gp_list_pilot[i].predict(x_test, return_std=True)
        print(f"  GP-{name} 预测：mu={mu[0]:.2f}, sigma={std[0]:.4f}")

    # ===== 测试 2: 耦合矩阵估计 =====
    print("\n" + "="*60)
    print("测试 2: 梯度外积耦合矩阵估计（3×3）")
    print("="*60)

    W_data = model.estimate_coupling_from_gradients(
        gp_list_pilot, fake_database, n_samples=10
    )

    print(f"\n数据驱动耦合矩阵 W_data (3×3):")
    print(W_data)
    print(f"  形状：{W_data.shape}")
    print(f"  对称性误差：{np.max(np.abs(W_data - W_data.T)):.2e}")
    print(f"  最小特征值：{np.min(np.linalg.eigvalsh(W_data)):.4e}")

    # ===== 测试 3: build_W_t（约束 C-2） =====
    print("\n" + "="*60)
    print("测试 3: build_W_t - 约束 C-2 实现")
    print("="*60)

    # 模拟 LLM 在 Touchpoint 1a 推断的三个耦合矩阵
    W_time = np.array([
        [1.0, 0.7, 0.3],
        [0.7, 1.0, 0.4],
        [0.3, 0.4, 1.0]
    ])
    W_temp = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.6],
        [0.2, 0.6, 1.0]
    ])
    W_aging = np.array([
        [1.0, 0.3, 0.5],
        [0.3, 1.0, 0.4],
        [0.5, 0.4, 1.0]
    ])

    # 当前 Tchebycheff 权重
    weights = np.array([0.4, 0.35, 0.25])

    W_t = model.build_W_t(weights, W_time, W_temp, W_aging)

    print(f"\nW^(t) = (0.4*W_time + 0.35*W_temp + 0.25*W_aging) / 1.0")
    print(f"W^(t) =\n{W_t}")
    print(f"  对称：{np.allclose(W_t, W_t.T)}")
    print(f"  对角线全 1: {np.allclose(np.diag(W_t), 1.0)}")
    print(f"  最小特征值：{np.min(np.linalg.eigvalsh(W_t)):.4e}")

    # ===== 测试 4: ARD 长度尺度设置 =====
    print("\n" + "="*60)
    print("测试 4: LLM 驱动 ARD 长度尺度")
    print("="*60)

    # 模拟敏感度排序：I1 最敏感 (rank=1), I2 最不敏感 (rank=3)
    ell_base = 0.5
    alpha_ell = 0.8
    d = 3
    ranks = np.array([1, 2, 3])  # I1, SOC1, I2 的排序
    length_scales = ell_base * (ranks / d) ** alpha_ell

    print(f"  排序：{dict(zip(['I1', 'SOC1', 'I2'], ranks))}")
    print(f"  长度尺度：{dict(zip(['I1', 'SOC1', 'I2'], [f'{l:.4f}' for l in length_scales]))}")

    model.set_ard_length_scales(length_scales)

    gp_list_ard = model.train_main_gp(fake_database, coupling_matrix=W_t)
    print(f"  OK 带 ARD 长度尺度的 Main GP 训练成功")

    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)
