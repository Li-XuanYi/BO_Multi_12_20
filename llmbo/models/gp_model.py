"""
多输出高斯过程模型模块
实现两阶段GP训练：Pilot GP + Main GP
"""

import numpy as np
from typing import List, Dict, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
import warnings

# 导入自定义模块
from models.kernels import CompositeKernel, CouplingKernel
from models.gradients import GPGradientComputer
# 导入config
from config import get_algorithm_param

warnings.filterwarnings('ignore', category=UserWarning)


class MOGPModel:
    """
    多目标高斯过程模型
    
    功能：
    1. Pilot GP训练（标准Matern核，用于推理耦合矩阵）
    2. Main GP训练（耦合核，用于采集优化）
    3. 耦合矩阵估计（从GP梯度）
    4. 数据变换（Log10 for aging）
    """
    
    def __init__(
        self,
        use_coupling: bool = True,
        gamma_init: float = 0.5,
        verbose: bool = False
    ):
        """
        初始化多目标GP模型
        
        参数：
            use_coupling: 是否使用物理耦合核
            gamma_init: 初始耦合强度
            verbose: 详细输出
        """
        self.use_coupling = use_coupling
        self.gamma = gamma_init
        self.verbose = verbose
        
        # ========== 修复：删除transformer ==========
        # 数据变换已在main.py中完成，不需要在此重复变换
        # self.transformer = DataTransformer(
        #     enable_log_aging=True,
        #     verbose=verbose
        # )
        # ==========================================
        
        # GP梯度计算器
        self.gradient_computer = GPGradientComputer(
            epsilon=1e-4,
            verbose=verbose
        )
        
        # 存储训练好的GP列表
        self.gp_list = None
        self.coupling_matrix = None
    
    def train_pilot_gp(self, database: List[Dict]) -> List[GaussianProcessRegressor]:
        """
        训练Pilot GP（用于推理耦合矩阵）
        
        特点：
        - 使用标准Matern核（不包含物理耦合）
        - 目的：提供干净的梯度信息
        
        参数：
            database: 评估历史（已经过main.py中的DataTransformer变换）
        
        返回：
            gp_list: [gp_time, gp_temp, gp_aging]
        """
        if self.verbose:
            print(f"\n  [训练Pilot GP]")
        
        # ========== 修复：不再重复变换 ==========
        # database已经由main.py的DataTransformer变换过，直接使用
        # transformed_db = self.transformer.fit_transform_database(database)  # ← 删除重复变换
        # =======================================
        
        # 提取有效数据
        valid_data = [r for r in database if r['valid']]  # ← 直接使用database
        
        if len(valid_data) < 3:
            raise ValueError(f"有效数据不足：{len(valid_data)} < 3")
        
        # 构建训练数据
        X_train = np.array([[
            r['params']['current1'],
            r['params']['switch_soc'],
            r['params']['current2']
        ] for r in valid_data])
        
        y_train_time = np.array([r['time'] for r in valid_data])
        y_train_temp = np.array([r['temp'] for r in valid_data])
        y_train_aging = np.array([r['aging'] for r in valid_data])  # 已经是Log空间
        
        if self.verbose:
            print(f"    训练数据: {len(valid_data)}个有效点")
            print(f"    Time范围: [{np.min(y_train_time):.1f}, {np.max(y_train_time):.1f}]")
            print(f"    Temp范围: [{np.min(y_train_temp):.1f}, {np.max(y_train_temp):.1f}]")
            print(f"    Aging范围(log10): [{np.min(y_train_aging):.2f}, {np.max(y_train_aging):.2f}]")
        
        # 标准Matern核
        kernel = C(1.0, constant_value_bounds=(1e-3, 1e3)) * \
                 Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        
        # 训练3个独立GP
        gp_list = []
        
        for obj_name, y_train in [('Time', y_train_time), 
                                    ('Temp', y_train_temp), 
                                    ('Aging', y_train_aging)]:
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=get_algorithm_param('gp', 'n_restarts_optimizer', 5),
                alpha=get_algorithm_param('gp', 'alpha', 1e-6),
                normalize_y=get_algorithm_param('gp', 'normalize_y', True),
                random_state=42
            )
            
            gp.fit(X_train, y_train)
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
        
        特点：
        - 使用耦合核（CompositeKernel）
        - 包含物理耦合信息
        
        参数：
            database: 评估历史（已经过main.py中的DataTransformer变换）
            coupling_matrix: 耦合矩阵（3x3）
        
        返回：
            gp_list: [gp_time, gp_temp, gp_aging]
        """
        if self.verbose:
            print(f"\n  [训练Main GP]")
        
        # ========== 修复：不再重复变换 ==========
        # database已经由main.py的DataTransformer变换过，直接使用
        # transformed_db = self.transformer.fit_transform_database(database)  # ← 删除重复变换
        # =======================================
        
        # 提取有效数据
        valid_data = [r for r in database if r['valid']]  # ← 直接使用database
        
        if len(valid_data) < 3:
            raise ValueError(f"有效数据不足：{len(valid_data)} < 3")
        
        # 构建训练数据
        X_train = np.array([[
            r['params']['current1'],
            r['params']['switch_soc'],
            r['params']['current2']
        ] for r in valid_data])
        
        y_train_time = np.array([r['time'] for r in valid_data])
        y_train_temp = np.array([r['temp'] for r in valid_data])
        y_train_aging = np.array([r['aging'] for r in valid_data])
        
        if self.verbose:
            print(f"    训练数据: {len(valid_data)}个有效点")
        
        # 创建核函数
        if self.use_coupling and coupling_matrix is not None:
            # 使用耦合核
            kernel = CompositeKernel(
                coupling_matrix=coupling_matrix,
                gamma=self.gamma
            )
            
            if self.verbose:
                print(f"    使用耦合核 (γ={self.gamma:.3f})")
        else:
            # 回退到标准Matern核
            kernel = C(1.0, constant_value_bounds=(1e-3, 1e3)) * \
                     Matern(nu=2.5, length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            
            if self.verbose:
                print(f"    使用标准Matern核")
        
        # 训练3个独立GP
        gp_list = []
        
        for obj_name, y_train in [('Time', y_train_time), 
                                    ('Temp', y_train_temp), 
                                    ('Aging', y_train_aging)]:
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=get_algorithm_param('gp', 'n_restarts_optimizer', 5),
                alpha=get_algorithm_param('gp', 'alpha', 1e-6),
                normalize_y=get_algorithm_param('gp', 'normalize_y', True),
                random_state=42
            )
            
            gp.fit(X_train, y_train)
            gp_list.append(gp)
            
            if self.verbose:
                print(f"    GP-{obj_name}: 训练完成")
        
        # 保存
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
        从GP梯度估计耦合矩阵
        
        方法：
        1. 在参数空间采样N个点
        2. 计算每个点的GP梯度（Jacobian）
        3. 计算梯度的外积平均
        4. 归一化到[0,1]
        
        参数：
            gp_list: Pilot GP列表
            database: 评估历史（用于采样参考）
            n_samples: 采样点数
        
        返回：
            W_data: 数据驱动的耦合矩阵 (3x3)
        """
        if self.verbose:
            print(f"\n  [估计数据驱动耦合矩阵]")
        
        # 提取有效数据
        valid_data = [r for r in database if r['valid']]
        
        if len(valid_data) < n_samples:
            n_samples = len(valid_data)
        
        # 采样：从有效点中随机选择
        indices = np.random.choice(len(valid_data), size=n_samples, replace=False)
        X_samples = np.array([[
            valid_data[i]['params']['current1'],
            valid_data[i]['params']['switch_soc'],
            valid_data[i]['params']['current2']
        ] for i in indices])
        
        if self.verbose:
            print(f"    采样点数: {n_samples}")
        
        # 从GP梯度估计耦合矩阵
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
        
        策略：
        1. 如果W_llm不可用，使用W_data
        2. 否则加权融合：W = α·W_data + (1-α)·W_llm
        3. 后处理：对称化、归一化
        
        参数：
            W_data: 数据驱动的耦合矩阵
            W_llm: LLM推理的耦合矩阵（可能为None）
            alpha: 数据驱动权重
        
        返回：
            W_final: 融合后的耦合矩阵
        """
        if W_llm is None:
            if self.verbose:
                print(f"\n  [耦合矩阵融合] LLM矩阵不可用，使用数据矩阵")
            return W_data
        
        if self.verbose:
            print(f"\n  [耦合矩阵融合] α={alpha:.2f} (数据:{alpha:.0%}, LLM:{1-alpha:.0%})")
        
        # 加权融合
        W_merged = alpha * W_data + (1 - alpha) * W_llm
        
        # 对称化
        W_merged = (W_merged + W_merged.T) / 2.0
        
        # 强制对角线为1
        np.fill_diagonal(W_merged, 1.0)
        
        # 归一化到[0, 1]（保持对角线为1）
        off_diagonal_max = np.max(np.abs(W_merged - np.eye(3)))
        if off_diagonal_max > 1.0:
            # 缩放非对角元素
            W_normalized = np.eye(3) + (W_merged - np.eye(3)) / off_diagonal_max
            W_merged = W_normalized
        
        # 确保非负
        W_merged = np.clip(W_merged, 0.0, 1.0)
        np.fill_diagonal(W_merged, 1.0)
        
        return W_merged
    
    def fit(self, database: List[Dict]):
        """
        完整的两阶段训练流程（兼容main.py的接口）
        
        参数：
            database: 评估历史
        """
        # 训练Pilot GP
        gp_pilot = self.train_pilot_gp(database)
        
        # 估计耦合矩阵
        W_data = self.estimate_coupling_from_gradients(gp_pilot, database, n_samples=10)
        
        # 训练Main GP
        self.train_main_gp(database, coupling_matrix=W_data)
    
    def set_llm_coupling_matrix(self, W_llm: np.ndarray):
        """
        设置LLM推理的耦合矩阵
        
        参数：
            W_llm: LLM推理的耦合矩阵 (3x3)
        """
        self.W_llm = W_llm
        
        if self.verbose:
            print(f"  [MOGPModel] LLM耦合矩阵已设置")
    
    def update_coupling_matrix(self, method: str = 'gradient'):
        """
        更新耦合矩阵（在BO迭代中）
        
        参数：
            method: 更新方法
        """
        if not self.use_coupling or self.gp_list is None:
            return
        
        if self.verbose:
            print(f"  [更新耦合矩阵] method={method}")
    
    def update_gamma(self, improvement_rate: float):
        """
        更新gamma参数（自适应）
        
        参数：
            improvement_rate: 改进率
        """
        # 简单实现：保持gamma不变
        pass
    
    def get_gp_list(self) -> List[GaussianProcessRegressor]:
        """
        返回GP列表
        
        返回：
            gp_list: [gp_time, gp_temp, gp_aging]
        """
        return self.gp_list if self.gp_list is not None else []


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    print("测试 MOGPModel...")
    
    # 创建虚拟数据库
    np.random.seed(42)
    fake_database = []
    
    for i in range(20):
        fake_database.append({
            'params': {
                'current1': np.random.uniform(3.0, 6.0),
                'switch_soc': np.random.uniform(0.3, 0.7),
                'current2': np.random.uniform(1.0, 4.0)
            },
            'time': np.random.randint(30, 100),
            'temp': np.random.uniform(300, 310),
            'aging': np.random.uniform(0.001, 0.05),
            'valid': True
        })
    
    # 初始化模型
    model = MOGPModel(use_coupling=True, verbose=True)
    
    # 测试Pilot GP
    print("\n" + "="*60)
    print("测试1: Pilot GP训练")
    print("="*60)
    
    gp_list_pilot = model.train_pilot_gp(fake_database)
    print(f"\n✓ Pilot GP训练成功，得到{len(gp_list_pilot)}个模型")
    
    # 测试耦合矩阵估计
    print("\n" + "="*60)
    print("测试2: 耦合矩阵估计")
    print("="*60)
    
    W_data = model.estimate_coupling_from_gradients(
        gp_list_pilot,
        fake_database,
        n_samples=10
    )
    
    print(f"\n数据驱动耦合矩阵:")
    print(W_data)
    
    # 测试Main GP
    print("\n" + "="*60)
    print("测试3: Main GP训练（带耦合核）")
    print("="*60)
    
    gp_list_main = model.train_main_gp(fake_database, coupling_matrix=W_data)
    print(f"\n✓ Main GP训练成功")
    
    # 测试矩阵融合
    print("\n" + "="*60)
    print("测试4: 矩阵融合")
    print("="*60)
    
    W_llm = np.array([
        [1.0, 0.7, 0.2],
        [0.7, 1.0, 0.3],
        [0.2, 0.3, 1.0]
    ])
    
    W_final = model.merge_coupling_matrices(W_data, W_llm, alpha=0.5)
    
    print(f"\n融合矩阵:")
    print(W_final)
    
    print("\n测试完成！")
