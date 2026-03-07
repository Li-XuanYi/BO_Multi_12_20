"""
多输出高斯过程模型模块
实现两阶段GP训练：Pilot GP + Main GP

修改记录：
- 测试代码从3D(switch_soc)修正为4D(current1, time1, current2, v_switch)
- merge_coupling_matrices增加维度兼容性检查
- 清理过时注释
"""

import numpy as np
from typing import List, Dict, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
import warnings

# 导入自定义模块
from models.kernels import CompositeKernel, CouplingKernel, ensure_psd
from models.gradients import GPGradientComputer
# 导入config
from config import get_algorithm_param

warnings.filterwarnings('ignore', category=UserWarning)


class MOGPModel:
    """
    多目标高斯过程模型（4D版本）
    
    功能：
    1. Pilot GP训练（标准ARD Matern核，用于推理耦合矩阵）
    2. Main GP训练（复合核 = ARD Matern + 物理耦合核）
    3. 耦合矩阵估计（从GP梯度外积）
    4. 耦合矩阵融合（数据驱动 + LLM推理）
    
    决策空间：[current1, time1, current2, v_switch]，d=4
    目标空间：[time, temp, aging]，m=3
    """
    
    def __init__(
        self,
        use_coupling: bool = True,
        gamma_init: float = 0.5,
        n_dims: int = 4,
        verbose: bool = False
    ):
        """
        初始化多目标GP模型
        
        参数：
            use_coupling: 是否使用物理耦合核
            gamma_init: 初始耦合强度
            n_dims: 决策空间维度（默认 4）
            verbose: 详细输出
        """
        self.use_coupling = use_coupling
        self.gamma = gamma_init
        self.n_dims = n_dims
        self.param_keys = ['current1', 'time1', 'current2', 'v_switch']
        self.ard_length_scales = None  # LLM 驱动的长度尺度
        self.verbose = verbose
        
        # GP梯度计算器
        self.gradient_computer = GPGradientComputer(
            epsilon=1e-4,
            verbose=verbose
        )
        
        # 存储训练好的GP列表
        self.gp_list = None
        self.coupling_matrix = None
        self.W_llm = None  # LLM推理的耦合矩阵（显式声明）
    
    def _extract_training_data(self, database: List[Dict]):
        """
        从数据库提取训练数据
        
        参数：
            database: legacy格式评估历史
        
        返回：
            X_train: (N, 4) 参数矩阵
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
        训练Pilot GP（用于推理耦合矩阵）
        
        特点：标准ARD Matern核，提供干净的梯度信息
        
        参数：
            database: 评估历史（已经过DataTransformer变换）
        
        返回：
            gp_list: [gp_time, gp_temp, gp_aging]
        """
        if self.verbose:
            print(f"\n  [训练Pilot GP]")
        
        X_train, y_dict, valid_data = self._extract_training_data(database)
        
        if self.verbose:
            print(f"    训练数据: {len(valid_data)}个有效点")
            print(f"    Time范围: [{np.min(y_dict['time']):.1f}, {np.max(y_dict['time']):.1f}]")
            print(f"    Temp范围: [{np.min(y_dict['temp']):.1f}, {np.max(y_dict['temp']):.1f}]")
            print(f"    Aging范围: [{np.min(y_dict['aging']):.4f}, {np.max(y_dict['aging']):.4f}]")
        
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
        训练Main GP（用于采集优化）
        
        特点：复合核 = ARD Matern + γ·物理耦合核，支持LLM长度尺度
        
        参数：
            database: 评估历史（已经过DataTransformer变换）
            coupling_matrix: 耦合矩阵 (4x4)
        
        返回：
            gp_list: [gp_time, gp_temp, gp_aging]
        """
        if self.verbose:
            print(f"\n  [训练Main GP]")
        
        X_train, y_dict, valid_data = self._extract_training_data(database)
        
        if self.verbose:
            print(f"    训练数据: {len(valid_data)}个有效点")
        
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
                    print(f"    已设置 ARD 长度尺度: {self.ard_length_scales}")
            
            if self.verbose:
                print(f"    使用耦合核 (γ={self.gamma:.3f})")
        else:
            kernel = C(1.0, constant_value_bounds=(1e-3, 1e3)) * \
                     Matern(nu=2.5, 
                            length_scale=np.ones(self.n_dims), 
                            length_scale_bounds=(1e-2, 1e2))
            if self.verbose:
                print(f"    使用标准ARD Matern核")
        
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
        从GP梯度估计耦合矩阵 W_data (4×4)
        
        方法：梯度外积平均 → 归一化 → PSD保证
        """
        if self.verbose:
            print(f"\n  [估计数据驱动耦合矩阵]")
        
        X_train, y_dict, valid_data = self._extract_training_data(database)
        
        if len(valid_data) < n_samples:
            n_samples = len(valid_data)
        
        indices = np.random.choice(len(valid_data), size=n_samples, replace=False)
        X_samples = X_train[indices]
        
        if self.verbose:
            print(f"    采样点数: {n_samples}")
        
        W_data = self.gradient_computer.estimate_coupling_matrix(
            gp_list=gp_list,
            X_samples=X_samples,
            method='outer_product'
        )
        
        if self.verbose:
            print(f"    耦合矩阵已估计")
        
        return W_data
    
    def merge_coupling_matrices(
        self,
        W_data: np.ndarray,
        W_llm: Optional[np.ndarray],
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        融合数据驱动和LLM推理的耦合矩阵
        
        公式：W_final = α·W_data + (1-α)·W_llm
        后处理：对称化 → PSD保证 → 对角线=1 → clip[0,1]
        
        参数：
            W_data: 数据驱动的耦合矩阵 (d×d)
            W_llm: LLM推理的耦合矩阵（可能为None，或维度不匹配）
            alpha: 数据驱动权重 [0,1]
        
        返回：
            W_final: 融合后的耦合矩阵 (d×d)
        """
        d = W_data.shape[0]
        
        # ========== 维度兼容性检查 ==========
        if W_llm is None:
            if self.verbose:
                print(f"\n  [耦合矩阵融合] LLM矩阵不可用，使用数据矩阵")
            return W_data
        
        if W_llm.shape != (d, d):
            if self.verbose:
                print(f"\n  [耦合矩阵融合] W_llm维度{W_llm.shape}与W_data{W_data.shape}不匹配，仅使用W_data")
            return W_data
        # =====================================
        
        if self.verbose:
            print(f"\n  [耦合矩阵融合] α={alpha:.2f} (数据:{alpha:.0%}, LLM:{1-alpha:.0%})")
        
        # 加权融合
        W_merged = alpha * W_data + (1 - alpha) * W_llm
        
        # 对称化
        W_merged = (W_merged + W_merged.T) / 2.0
        
        # 强制对角线为1
        np.fill_diagonal(W_merged, 1.0)
        
        # PSD保证
        eps_psd = get_algorithm_param('composite_kernel', 'eps_psd', 1e-6)
        W_merged = ensure_psd(W_merged, eps=eps_psd)
        
        # 再次强制对角线为1 + 归一化到[0, 1]
        np.fill_diagonal(W_merged, 1.0)
        W_merged = np.clip(W_merged, 0.0, 1.0)
        np.fill_diagonal(W_merged, 1.0)
        
        return W_merged
    
    def fit(self, database: List[Dict]):
        """
        完整的两阶段训练流程（便捷接口）
        """
        gp_pilot = self.train_pilot_gp(database)
        W_data = self.estimate_coupling_from_gradients(gp_pilot, database, n_samples=10)
        self.train_main_gp(database, coupling_matrix=W_data)
    
    def set_llm_coupling_matrix(self, W_llm: np.ndarray):
        """设置LLM推理的耦合矩阵 (4×4)"""
        self.W_llm = W_llm
        if self.verbose:
            print(f"  [MOGPModel] LLM耦合矩阵已设置")
    
    def set_ard_length_scales(self, length_scales):
        """
        设置ARD长度尺度（LLM驱动）
        
        参数：
            length_scales: np.ndarray, 形状 (d,)
        """
        length_scales = np.asarray(length_scales, dtype=float)
        assert len(length_scales) == self.n_dims, \
            f"Length scales维度不匹配: 期望{self.n_dims}, 得到{len(length_scales)}"
        assert np.all(length_scales > 0), "All length scales must be > 0"
        
        self.ard_length_scales = length_scales.copy()
        
        if self.verbose:
            print(f"  [MOGPModel] ARD长度尺度已设置: {self.ard_length_scales}")
    
    def update_coupling_matrix(self, method: str = 'gradient'):
        """更新耦合矩阵（占位，由main.py外部调用实现）"""
        if not self.use_coupling or self.gp_list is None:
            return
        if self.verbose:
            print(f"  [更新耦合矩阵] method={method}")
    
    def update_gamma(self, improvement_rate: float):
        """更新gamma参数（由main.py的HV自适应外部处理）"""
        pass
    
    def get_gp_list(self) -> List[GaussianProcessRegressor]:
        """返回当前GP列表 [gp_time, gp_temp, gp_aging]"""
        return self.gp_list if self.gp_list is not None else []


# ============================================================
# 快速测试（4D版本）
# ============================================================
if __name__ == "__main__":
    print("测试 MOGPModel（4D参数空间）...")
    
    # ===== 创建4D虚拟数据库 =====
    np.random.seed(42)
    fake_database = []
    
    for i in range(20):
        fake_database.append({
            'params': {
                'current1': np.random.uniform(3.0, 6.0),
                'time1': np.random.uniform(2.0, 40.0),
                'current2': np.random.uniform(1.0, 4.0),
                'v_switch': np.random.uniform(3.8, 4.2)
            },
            'time': np.random.uniform(600, 3600),   # 秒
            'temp': np.random.uniform(300, 315),     # K
            'aging': np.random.uniform(-5.0, -1.5),  # log10空间
            'valid': True
        })
    
    # 初始化模型
    model = MOGPModel(use_coupling=True, verbose=True)
    
    # ===== 测试1: Pilot GP训练 =====
    print("\n" + "="*60)
    print("测试1: Pilot GP训练（4D ARD Matern）")
    print("="*60)
    
    gp_list_pilot = model.train_pilot_gp(fake_database)
    print(f"\n✓ Pilot GP训练成功，得到{len(gp_list_pilot)}个模型")
    
    # 验证预测
    x_test = np.array([[4.5, 20.0, 2.5, 4.0]])
    for i, name in enumerate(['Time', 'Temp', 'Aging']):
        mu, std = gp_list_pilot[i].predict(x_test, return_std=True)
        print(f"  GP-{name} 预测: μ={mu[0]:.2f}, σ={std[0]:.4f}")
    
    # ===== 测试2: 耦合矩阵估计 =====
    print("\n" + "="*60)
    print("测试2: 梯度外积耦合矩阵估计（4×4）")
    print("="*60)
    
    W_data = model.estimate_coupling_from_gradients(
        gp_list_pilot, fake_database, n_samples=10
    )
    
    print(f"\n数据驱动耦合矩阵 W_data (4×4):")
    print(W_data)
    print(f"  形状: {W_data.shape}")
    print(f"  对称性误差: {np.max(np.abs(W_data - W_data.T)):.2e}")
    print(f"  最小特征值: {np.min(np.linalg.eigvalsh(W_data)):.4e}")
    
    # ===== 测试3: Main GP训练 =====
    print("\n" + "="*60)
    print("测试3: Main GP训练（复合核 = ARD + 耦合）")
    print("="*60)
    
    gp_list_main = model.train_main_gp(fake_database, coupling_matrix=W_data)
    print(f"\n✓ Main GP训练成功")
    
    for i, name in enumerate(['Time', 'Temp', 'Aging']):
        mu, std = gp_list_main[i].predict(x_test, return_std=True)
        print(f"  GP-{name} 预测: μ={mu[0]:.2f}, σ={std[0]:.4f}")
    
    # ===== 测试4: 矩阵融合（4×4） =====
    print("\n" + "="*60)
    print("测试4: W_data + W_llm 融合（4×4）")
    print("="*60)
    
    # 模拟LLM推理的4×4耦合矩阵
    # 物理含义：I1↔T1强耦合(0.7), I1↔I2中等(0.4), I2↔V_sw中等(0.5), 其余弱
    W_llm = np.array([
        [1.0, 0.7, 0.4, 0.2],
        [0.7, 1.0, 0.3, 0.3],
        [0.4, 0.3, 1.0, 0.5],
        [0.2, 0.3, 0.5, 1.0]
    ])
    
    W_final = model.merge_coupling_matrices(W_data, W_llm, alpha=0.5)
    
    print(f"\n融合矩阵 W_final (4×4):")
    print(W_final)
    print(f"  对称: {np.allclose(W_final, W_final.T)}")
    print(f"  对角线全1: {np.allclose(np.diag(W_final), 1.0)}")
    print(f"  值域 [0,1]: [{np.min(W_final):.3f}, {np.max(W_final):.3f}]")
    print(f"  最小特征值: {np.min(np.linalg.eigvalsh(W_final)):.4e}")
    
    # ===== 测试5: 维度不匹配容错 =====
    print("\n" + "="*60)
    print("测试5: W_llm维度不匹配容错")
    print("="*60)
    
    W_llm_wrong = np.array([
        [1.0, 0.7, 0.2],
        [0.7, 1.0, 0.3],
        [0.2, 0.3, 1.0]
    ])
    
    W_fallback = model.merge_coupling_matrices(W_data, W_llm_wrong, alpha=0.5)
    print(f"  W_llm(3×3) + W_data(4×4) → 回退到W_data: {np.allclose(W_fallback, W_data)}")
    
    W_none = model.merge_coupling_matrices(W_data, None, alpha=0.5)
    print(f"  W_llm=None → 回退到W_data: {np.allclose(W_none, W_data)}")
    
    # ===== 测试6: ARD长度尺度设置 =====
    print("\n" + "="*60)
    print("测试6: LLM驱动ARD长度尺度")
    print("="*60)
    
    # 模拟敏感度排序：I1最敏感(rank=1), V_sw最不敏感(rank=4)
    ell_base = 0.5
    alpha_ell = 0.8
    d = 4
    ranks = np.array([1, 3, 2, 4])  # I1, T1, I2, V_sw的排序
    length_scales = ell_base * (ranks / d) ** alpha_ell
    
    print(f"  排序: {dict(zip(['I1','T1','I2','V_sw'], ranks))}")
    print(f"  长度尺度: {dict(zip(['I1','T1','I2','V_sw'], [f'{l:.4f}' for l in length_scales]))}")
    
    model.set_ard_length_scales(length_scales)
    
    gp_list_ard = model.train_main_gp(fake_database, coupling_matrix=W_data)
    print(f"  ✓ 带ARD长度尺度的Main GP训练成功")
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)