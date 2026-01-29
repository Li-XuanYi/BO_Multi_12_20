"""
数据变换模块
处理不同量级的目标值（Input Warping）
"""

import numpy as np
from typing import Dict, List, Tuple


class DataTransformer:
    """
    数据变换器
    
    功能：
    1. Log10变换（处理小数值，如aging ~1e-5）
    2. Min-Max归一化（统一量纲）
    3. 可逆变换（预测后还原）
    """
    
    def __init__(self, enable_log_aging: bool = True, verbose: bool = False):
        """
        初始化变换器
        
        参数：
            enable_log_aging: 是否对aging进行Log10变换
            verbose: 详细输出
        """
        self.enable_log_aging = enable_log_aging
        self.verbose = verbose
        
        # 缓存统计量（用于归一化）
        self.stats = {
            'time': {'mean': None, 'std': None, 'min': None, 'max': None},
            'temp': {'mean': None, 'std': None, 'min': None, 'max': None},
            'aging': {'mean': None, 'std': None, 'min': None, 'max': None}
        }
    
    def fit_transform_database(self, database: List[Dict]) -> List[Dict]:
        """
        拟合变换器并转换数据库
        
        参数：
            database: 原始评估历史
        
        返回：
            transformed_database: 变换后的数据库
        """
        # 提取有效数据
        valid_data = [r for r in database if r['valid']]
        
        if len(valid_data) == 0:
            return database
        
        # 提取目标值
        times = np.array([r['time'] for r in valid_data])
        temps = np.array([r['temp'] for r in valid_data])
        agings = np.array([r['aging'] for r in valid_data])
        
        # 1. 对aging进行Log10变换
        if self.enable_log_aging:
            # ========== 修复：严格处理异常值 ==========
            agings_safe = np.clip(agings, 1e-6, None)
            agings_safe = np.nan_to_num(agings_safe, nan=1e-6, posinf=0.1, neginf=1e-6)
            agings_transformed = np.log10(agings_safe)
            
            if np.any(np.isnan(agings_transformed)):
                agings_transformed = np.nan_to_num(agings_transformed, nan=-5.0)
            # ==========================================
        else:
            agings_transformed = agings
        
        # 2. 计算统计量（用于归一化）
        self.stats['time'] = {
            'mean': np.mean(times),
            'std': np.std(times) + 1e-6,
            'min': np.min(times),
            'max': np.max(times)
        }
        self.stats['temp'] = {
            'mean': np.mean(temps),
            'std': np.std(temps) + 1e-6,
            'min': np.min(temps),
            'max': np.max(temps)
        }
        self.stats['aging'] = {
            'mean': np.mean(agings_transformed),
            'std': np.std(agings_transformed) + 1e-6,
            'min': np.min(agings_transformed),
            'max': np.max(agings_transformed)
        }
        
        if self.verbose:
            print(f"  [DataTransformer] 统计量:")
            print(f"    Time: mean={self.stats['time']['mean']:.1f}, std={self.stats['time']['std']:.1f}")
            print(f"    Temp: mean={self.stats['temp']['mean']:.2f}, std={self.stats['temp']['std']:.2f}")
            if self.enable_log_aging:
                print(f"    Aging (log10): mean={self.stats['aging']['mean']:.2f}, std={self.stats['aging']['std']:.2f}")
            else:
                print(f"    Aging: mean={self.stats['aging']['mean']:.4f}, std={self.stats['aging']['std']:.4f}")
        
        # 3. 转换数据库
        transformed_database = []
        for record in database:
            if not record['valid']:
                transformed_database.append(record)
                continue
            
            aging_raw = record['aging']
            
            if self.enable_log_aging:
                # ========== 修复：单个值严格检查 ==========
                aging_safe = max(aging_raw, 1e-6)
                
                if np.isnan(aging_safe) or np.isinf(aging_safe):
                    aging_safe = 1e-6
                
                aging_transformed = np.log10(aging_safe)
                
                if np.isnan(aging_transformed) or np.isinf(aging_transformed):
                    aging_transformed = -5.0
                # ==========================================
            else:
                aging_transformed = aging_raw
            
            transformed_record = record.copy()
            transformed_record['aging_raw'] = aging_raw  # 保存原始值
            transformed_record['aging'] = aging_transformed
            
            transformed_database.append(transformed_record)
        
        return transformed_database
    
    def transform_objectives(self, objectives: np.ndarray) -> np.ndarray:
        """
        变换目标值（用于预测）
        
        参数：
            objectives: [time, temp, aging]
        
        返回：
            transformed: 变换后的目标值
        """
        transformed = objectives.copy()
        
        if self.enable_log_aging:
            transformed[2] = np.log10(objectives[2] + 1e-10)
        
        return transformed
    
    def inverse_transform_objectives(self, transformed: np.ndarray) -> np.ndarray:
        """
        逆变换（还原原始尺度）
        
        参数：
            transformed: 变换后的目标值
        
        返回：
            objectives: 原始尺度的目标值
        """
        objectives = transformed.copy()
        
        if self.enable_log_aging:
            objectives[2] = 10**transformed[2] - 1e-10
            objectives[2] = np.maximum(objectives[2], 0.0)  # 避免负值
        
        return objectives
    
    def standardize(self, objectives: np.ndarray) -> np.ndarray:
        """
        标准化（Z-score）
        
        参数：
            objectives: [time, temp, aging] (已变换)
        
        返回：
            standardized: 标准化后的值
        """
        standardized = np.zeros(3)
        
        standardized[0] = (objectives[0] - self.stats['time']['mean']) / self.stats['time']['std']
        standardized[1] = (objectives[1] - self.stats['temp']['mean']) / self.stats['temp']['std']
        standardized[2] = (objectives[2] - self.stats['aging']['mean']) / self.stats['aging']['std']
        
        return standardized
    
    def get_transformed_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        获取变换后的边界（用于Tchebycheff标量化）
        
        返回：
            bounds: {'ideal': [...], 'reference': [...]}
        """
        if self.stats['time']['mean'] is None:
            # 未拟合，返回默认值（注意：aging已经是Log空间）
            if self.enable_log_aging:
                # Log空间的默认边界
                ideal_aging = np.log10(1e-6 + 1e-10)      # ≈ -6.0
                reference_aging = np.log10(0.1 + 1e-10)   # ≈ -1.0
            else:
                ideal_aging = 0.0
                reference_aging = 0.1
            
            return {
                'ideal': np.array([10, 298.15, ideal_aging]),
                'reference': np.array([300, 309.0, reference_aging])
            }
        
        # 基于数据的边界
        ideal = np.array([
            self.stats['time']['min'],
            self.stats['temp']['min'],
            self.stats['aging']['min']
        ])
        
        reference = np.array([
            self.stats['time']['max'],
            self.stats['temp']['max'],
            self.stats['aging']['max']
        ])
        
        return {'ideal': ideal, 'reference': reference}


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    print("测试 DataTransformer...")
    
    # 创建虚拟数据库
    np.random.seed(42)
    database = []
    
    for i in range(20):
        database.append({
            'params': {'current1': 4.5, 'switch_soc': 0.5, 'current2': 2.5},
            'time': 50 + np.random.randn() * 10,
            'temp': 305 + np.random.randn() * 2,
            'aging': 0.02 + np.random.randn() * 0.005,  # 小数值
            'valid': True
        })
    
    # 初始化变换器
    transformer = DataTransformer(enable_log_aging=True, verbose=True)
    
    # 拟合并变换
    transformed_db = transformer.fit_transform_database(database)
    
    print(f"\n原始aging范围: [{min([r['aging_raw'] for r in transformed_db]):.4f}, "
          f"{max([r['aging_raw'] for r in transformed_db]):.4f}]")
    print(f"变换后aging范围: [{min([r['aging'] for r in transformed_db]):.2f}, "
          f"{max([r['aging'] for r in transformed_db]):.2f}]")
    
    # 测试逆变换
    test_objectives = np.array([50.0, 305.0, -1.7])  # log10(0.02) ≈ -1.7
    restored = transformer.inverse_transform_objectives(test_objectives)
    print(f"\n逆变换测试:")
    print(f"  输入 (log10): {test_objectives}")
    print(f"  还原: {restored}")
    
    print("\n测试完成！")