"""
ExperimentDatabase独立模块
三层数据库结构

修改记录：
- get_pareto_front: 非支配排序前对目标值归一化，避免量级差异导致aging维度被忽略
- 新增 get_experiment_count / get_statistics 便捷查询
- _hv_2d: 修复sweepline需要非支配子前沿的问题
- compute_hypervolume: 增加空输入保护
"""

import sqlite3
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional


class ExperimentDatabase:
    """
    三层数据库结构：
    - Table A: experiments (不可变物理仿真记录)
    - Table B: states (每轮迭代状态快照)
    - Table C: llm_logs (LLM调用日志)
    """
    def __init__(self, db_path: str = ':memory:'):
        """
        参数:
            db_path: 数据库路径 (':memory:' 表示内存数据库)
        """
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
    
    def _create_tables(self):
        """创建三个表"""
        cursor = self.conn.cursor()
        
        # Table A: experiments — 物理仿真记录（不可变事实）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                current1 REAL,
                time1 REAL,
                current2 REAL,
                v_switch REAL,
                time REAL,
                temp REAL,
                aging REAL,
                valid INTEGER,
                violation TEXT,
                rationale TEXT,
                scenario TEXT
            )
        ''')
        
        # Table B: states — 每轮迭代的优化状态快照
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS states (
                iteration INTEGER PRIMARY KEY,
                timestamp TEXT,
                weights TEXT,
                gamma REAL,
                hypervolume REAL,
                n_pareto INTEGER,
                llm_focus_mu TEXT,
                llm_focus_sigma TEXT
            )
        ''')
        
        # Table C: llm_logs — LLM推理调用日志
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS llm_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                call_type TEXT,
                model TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                latency REAL,
                result TEXT
            )
        ''')
        
        self.conn.commit()
    
    # ================================================================
    # 写入接口
    # ================================================================
    
    def add_experiment(self, params: Dict, result: Dict, meta: Dict = None):
        """
        添加实验记录
        
        参数:
            params: {'current1': ..., 'time1': ..., 'current2': ..., 'v_switch': ...}
            result: {'time': ..., 'temp': ..., 'aging': ..., 'valid': ..., 'violation': ...}
            meta: {'rationale': ..., 'scenario': ...} (可选)
        """
        cursor = self.conn.cursor()
        meta = meta or {}
        
        cursor.execute('''
            INSERT INTO experiments (
                timestamp, current1, time1, current2, v_switch,
                time, temp, aging, valid, violation, rationale, scenario
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            params['current1'], params['time1'], params['current2'], params['v_switch'],
            result['time'], result['temp'], result['aging'],
            int(result['valid']), result.get('violation', ''),
            meta.get('rationale', ''), meta.get('scenario', '')
        ))
        self.conn.commit()
    
    def add_state(self, iteration: int, state: Dict):
        """
        添加/更新状态快照
        
        参数:
            iteration: 迭代轮数
            state: {'weights': [...], 'gamma': ..., 'hypervolume': ..., ...}
        """
        cursor = self.conn.cursor()
        
        # 安全序列化：处理 np.ndarray / list / dict
        def _safe_json(val):
            if isinstance(val, np.ndarray):
                return json.dumps(val.tolist())
            elif isinstance(val, (list, dict)):
                return json.dumps(val)
            else:
                return json.dumps(val)
        
        cursor.execute('''
            INSERT OR REPLACE INTO states (
                iteration, timestamp, weights, gamma, hypervolume, n_pareto,
                llm_focus_mu, llm_focus_sigma
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            iteration,
            datetime.now().isoformat(),
            _safe_json(state.get('weights', [])),
            state.get('gamma', 0.0),
            state.get('hypervolume', 0.0),
            state.get('n_pareto', 0),
            _safe_json(state.get('llm_focus_mu', {})),
            _safe_json(state.get('llm_focus_sigma', {}))
        ))
        self.conn.commit()
    
    def add_llm_log(self, call_type: str, model: str, tokens: Dict, latency: float, result: str):
        """
        添加LLM调用日志
        
        参数:
            call_type: 'warmstart' | 'coupling' | 'weighting' | 'sensitivity' | 'acquisition'
            model: 模型名称
            tokens: {'prompt': ..., 'completion': ...}
            latency: 延迟(秒)
            result: 返回结果(JSON字符串)
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO llm_logs (
                timestamp, call_type, model, prompt_tokens, completion_tokens,
                latency, result
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            call_type, model,
            tokens.get('prompt', 0), tokens.get('completion', 0),
            latency, result
        ))
        self.conn.commit()
    
    # ================================================================
    # 读取接口
    # ================================================================
    
    def get_all_experiments(self) -> List[Dict]:
        """获取所有实验记录"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM experiments')
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]
    
    def get_valid_experiments(self) -> List[Dict]:
        """获取所有有效实验"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM experiments WHERE valid = 1')
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]
    
    def get_experiment_count(self) -> Dict[str, int]:
        """获取实验计数统计"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM experiments')
        total = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM experiments WHERE valid = 1')
        valid = cursor.fetchone()[0]
        return {'total': total, 'valid': valid, 'invalid': total - valid}
    
    def get_pareto_front(self) -> List[Dict]:
        """
        提取Pareto前沿（归一化非支配排序）
        
        关键修正：先对三个目标进行min-max归一化再做支配比较，
        避免量级差异（time~3000, temp~310, aging~0.01）导致
        aging维度在支配判断中被忽略。
        
        返回的是原始（未归一化）的实验记录。
        """
        valid_data = self.get_valid_experiments()
        
        if len(valid_data) == 0:
            return []
        
        # 提取原始目标值
        objectives = np.array([
            [r['time'], r['temp'], r['aging']] for r in valid_data
        ])
        
        # ========== 归一化后做非支配排序 ==========
        obj_min = objectives.min(axis=0)
        obj_max = objectives.max(axis=0)
        obj_range = obj_max - obj_min
        # 避免除零：如果某维度所有值相同，归一化后该维度全为0
        obj_range = np.where(obj_range > 1e-12, obj_range, 1.0)
        objectives_norm = (objectives - obj_min) / obj_range
        # ==========================================
        
        n = len(valid_data)
        pareto_mask = np.ones(n, dtype=bool)
        
        for i in range(n):
            if not pareto_mask[i]:
                continue
            for j in range(n):
                if i == j or not pareto_mask[j]:
                    continue
                # j 支配 i ？（在归一化空间中判断）
                if (np.all(objectives_norm[j] <= objectives_norm[i]) and
                        np.any(objectives_norm[j] < objectives_norm[i])):
                    pareto_mask[i] = False
                    break
        
        pareto_indices = np.where(pareto_mask)[0]
        return [valid_data[i] for i in pareto_indices]
    
    def to_legacy_format(self) -> List[Dict]:
        """
        转换为旧版格式（兼容GP训练、可视化等下游模块）
        
        返回:
            [{'params': {...}, 'time': ..., 'temp': ..., 'aging': ..., 'valid': ...}, ...]
        """
        experiments = self.get_all_experiments()
        
        legacy_db = []
        for exp in experiments:
            legacy_db.append({
                'params': {
                    'current1': exp['current1'],
                    'time1': exp['time1'],
                    'current2': exp['current2'],
                    'v_switch': exp['v_switch']
                },
                'time': exp['time'],
                'temp': exp['temp'],
                'aging': exp['aging'],
                'valid': bool(exp['valid']),
                'violation': exp['violation'],
                'rationale': exp.get('rationale', ''),
                'scenario': exp.get('scenario', '')
            })
        
        return legacy_db
    
    def get_hv_history(self) -> List[float]:
        """获取HV历史（按迭代顺序）"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT hypervolume FROM states ORDER BY iteration')
        rows = cursor.fetchall()
        return [row[0] for row in rows]
    
    def get_statistics(self) -> Dict:
        """获取汇总统计（用于日志/调试）"""
        valid = self.get_valid_experiments()
        pareto = self.get_pareto_front()
        counts = self.get_experiment_count()
        
        stats = {
            'n_total': counts['total'],
            'n_valid': counts['valid'],
            'n_invalid': counts['invalid'],
            'n_pareto': len(pareto),
        }
        
        if len(valid) > 0:
            times = [r['time'] for r in valid]
            temps = [r['temp'] for r in valid]
            agings = [r['aging'] for r in valid]
            stats['time_range'] = (min(times), max(times))
            stats['temp_range'] = (min(temps), max(temps))
            stats['aging_range'] = (min(agings), max(agings))
        
        return stats
    
    # ================================================================
    # 持久化
    # ================================================================
    
    def save(self, filepath: str):
        """保存数据库到文件"""
        disk_conn = sqlite3.connect(filepath)
        self.conn.backup(disk_conn)
        disk_conn.close()
    
    def close(self):
        """关闭数据库"""
        self.conn.close()


# ============================================================
# Hypervolume计算函数
# ============================================================
def compute_hypervolume(pareto_front: np.ndarray, reference_point: np.ndarray) -> float:
    """
    计算Hypervolume指标（最小化问题）
    
    参数:
        pareto_front: (N, M) Pareto前沿目标值（原始空间）
        reference_point: (M,) 参考点（最坏情况边界）
    
    返回:
        hv: Hypervolume值（越大越好，表示Pareto前沿质量越高）
    """
    if pareto_front is None or len(pareto_front) == 0:
        return 0.0
    
    pareto_front = np.atleast_2d(pareto_front)
    
    if pareto_front.shape[0] == 0:
        return 0.0
    
    # 尝试使用pymoo（精确、高效）
    try:
        from pymoo.indicators.hv import HV
        ind = HV(ref_point=reference_point.astype(float))
        hv = ind(pareto_front.astype(float))
        return float(hv)
    except ImportError:
        pass
    
    # Fallback: 手写精确计算
    m = pareto_front.shape[1]
    if m == 3:
        return _hv_3d_exact(pareto_front, reference_point)
    elif m == 2:
        return _hv_2d(pareto_front, reference_point)
    else:
        # 高维需要pymoo，这里返回0并警告
        import warnings
        warnings.warn(f"Hypervolume for {m}D requires pymoo. Returning 0.0.")
        return 0.0


def compute_hypervolume_normalized(
    pareto_front: np.ndarray,
    reference_point: np.ndarray,
    ideal_point: np.ndarray
) -> float:
    """
    计算归一化Hypervolume（值域 [0, 1]）
    
    先将目标值归一化到 [0, 1]（使用 ideal 和 reference），
    再在归一化空间中计算 HV。
    
    参数:
        pareto_front: (N, M) Pareto前沿目标值（原始空间）
        reference_point: (M,) 参考点（最坏情况）
        ideal_point: (M,) 理想点（最好情况）
    
    返回:
        hv_normalized: [0, 1] 归一化HV
    """
    if pareto_front is None or len(pareto_front) == 0:
        return 0.0
    
    pareto_front = np.atleast_2d(pareto_front)
    if pareto_front.shape[0] == 0:
        return 0.0
    
    # 归一化范围
    range_val = reference_point - ideal_point
    range_val = np.where(np.abs(range_val) > 1e-12, range_val, 1.0)
    
    # 归一化 Pareto 前沿
    pf_norm = (pareto_front - ideal_point) / range_val
    ref_norm = np.ones(len(reference_point))  # 归一化后参考点为 [1, 1, ..., 1]
    
    return compute_hypervolume(pf_norm, ref_norm)


def _hv_3d_exact(pareto_front: np.ndarray, ref: np.ndarray) -> float:
    """
    3D Hypervolume精确计算（简化WFG sweepline）
    
    对每个Pareto点，计算其在第一维度上的独占切片宽度，
    乘以该切片内后两维的2D HV贡献。
    """
    # 过滤：只保留被参考点严格支配的点
    valid_mask = np.all(pareto_front < ref, axis=1)
    valid_front = pareto_front[valid_mask]
    
    if len(valid_front) == 0:
        return 0.0
    
    # 按第一维降序排列（从大到小，sweepline从ref向原点扫）
    sorted_indices = np.argsort(-valid_front[:, 0])
    sorted_front = valid_front[sorted_indices]
    
    hv = 0.0
    n = len(sorted_front)
    
    for i in range(n):
        point = sorted_front[i]
        
        # 第一维切片宽度
        if i == 0:
            x_hi = ref[0]
        else:
            x_hi = sorted_front[i - 1, 0]
        x_lo = point[0]
        x_width = x_hi - x_lo
        
        if x_width <= 0:
            continue
        
        # 该切片内的后两维子前沿 = sorted_front[0:i+1] 在 dim[1:3]
        # 因为是按dim0降序排的，前 i+1 个点都在该切片的"累积"范围内
        sub_front_all = sorted_front[:i + 1, 1:3]
        sub_ref = ref[1:3]
        
        # 过滤无效点
        sub_valid = np.all(sub_front_all < sub_ref, axis=1)
        sub_front = sub_front_all[sub_valid]
        
        if len(sub_front) > 0:
            # 提取子前沿的非支配点
            sub_front_nd = _extract_2d_nondominated(sub_front)
            hv_2d = _hv_2d(sub_front_nd, sub_ref)
            hv += x_width * hv_2d
    
    return hv


def _extract_2d_nondominated(points: np.ndarray) -> np.ndarray:
    """
    提取2D点集的非支配子集
    
    参数：
        points: (N, 2) 目标值
    
    返回：
        nd_points: (K, 2) 非支配点
    """
    if len(points) <= 1:
        return points
    
    # 按第一维排序
    sorted_idx = np.argsort(points[:, 0])
    sorted_pts = points[sorted_idx]
    
    # 扫描：维护第二维的当前最小值
    nd_list = [sorted_pts[0]]
    min_y = sorted_pts[0, 1]
    
    for i in range(1, len(sorted_pts)):
        if sorted_pts[i, 1] < min_y:
            nd_list.append(sorted_pts[i])
            min_y = sorted_pts[i, 1]
    
    return np.array(nd_list)


def _hv_2d(pareto_front: np.ndarray, ref: np.ndarray) -> float:
    """
    2D Hypervolume精确计算（sweepline）
    
    参数:
        pareto_front: (N, 2)
        ref: (2,) 参考点
    
    返回:
        hv: 面积
    """
    # 过滤不在参考点内的点
    valid_mask = np.all(pareto_front < ref, axis=1)
    valid_front = pareto_front[valid_mask]
    
    if len(valid_front) == 0:
        return 0.0
    
    # 提取非支配子集并按dim0排序
    nd_front = _extract_2d_nondominated(valid_front)
    nd_sorted = nd_front[np.argsort(nd_front[:, 0])]
    
    hv = 0.0
    prev_x = ref[0]
    for i in range(len(nd_sorted) - 1, -1, -1):
        x_i, y_i = nd_sorted[i]
        width = prev_x - x_i
        height = ref[1] - y_i
        hv += width * height
        prev_x = x_i
    
    return hv


# ============================================================
# 自测
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 ExperimentDatabase + Hypervolume")
    print("=" * 60)
    
    # ===== 测试1: 基本CRUD =====
    print("\n[测试1] 基本CRUD操作")
    db = ExperimentDatabase(':memory:')
    
    # 添加实验
    for i in range(10):
        params = {
            'current1': 3.0 + i * 0.3,
            'time1': 5.0 + i * 3.0,
            'current2': 1.5 + i * 0.2,
            'v_switch': 3.9 + i * 0.03
        }
        result = {
            'time': 1000 + i * 200,
            'temp': 300 + i * 1.5,
            'aging': 0.005 + i * 0.005,
            'valid': i != 3,  # 第3个无效
            'violation': 'temp exceeded' if i == 3 else ''
        }
        meta = {
            'rationale': f'test strategy {i}',
            'scenario': 'balanced' if i % 2 == 0 else 'aggressive'
        }
        db.add_experiment(params, result, meta)
    
    counts = db.get_experiment_count()
    print(f"  总实验: {counts['total']}, 有效: {counts['valid']}, 无效: {counts['invalid']}")
    assert counts['total'] == 10
    assert counts['valid'] == 9
    assert counts['invalid'] == 1
    print("  ✓ CRUD通过")
    
    # ===== 测试2: Pareto前沿（归一化排序） =====
    print("\n[测试2] Pareto前沿提取（归一化非支配排序）")
    pareto = db.get_pareto_front()
    print(f"  Pareto前沿: {len(pareto)}个解")
    
    # 验证：前沿中的点不应被其他前沿点支配
    for i, p1 in enumerate(pareto):
        for j, p2 in enumerate(pareto):
            if i != j:
                obj1 = np.array([p1['time'], p1['temp'], p1['aging']])
                obj2 = np.array([p2['time'], p2['temp'], p2['aging']])
                assert not (np.all(obj2 <= obj1) and np.any(obj2 < obj1)), \
                    f"Pareto点{i}被点{j}支配！"
    print("  ✓ 非支配性验证通过")
    
    # ===== 测试3: 量级差异场景 =====
    print("\n[测试3] 量级差异场景（aging应参与区分）")
    db2 = ExperimentDatabase(':memory:')
    
    # 构造：两个点time和temp相同，但aging不同
    db2.add_experiment(
        {'current1': 4.0, 'time1': 10.0, 'current2': 2.0, 'v_switch': 4.0},
        {'time': 2000.0, 'temp': 305.0, 'aging': 0.01, 'valid': True}
    )
    db2.add_experiment(
        {'current1': 4.5, 'time1': 12.0, 'current2': 2.5, 'v_switch': 4.0},
        {'time': 2000.0, 'temp': 305.0, 'aging': 0.05, 'valid': True}
    )
    
    pareto2 = db2.get_pareto_front()
    # 点A(aging=0.01)应支配点B(aging=0.05)，因为time和temp相同
    assert len(pareto2) == 1, f"期望1个Pareto点，得到{len(pareto2)}"
    assert pareto2[0]['aging'] == 0.01, "应选择aging更小的点"
    print(f"  ✓ aging区分正确：Pareto={len(pareto2)}个解，aging={pareto2[0]['aging']}")
    
    # ===== 测试4: Legacy格式 =====
    print("\n[测试4] Legacy格式转换")
    legacy = db.to_legacy_format()
    assert len(legacy) == 10
    assert 'params' in legacy[0]
    assert 'current1' in legacy[0]['params']
    assert 'v_switch' in legacy[0]['params']
    print(f"  ✓ Legacy格式正确，{len(legacy)}条记录")
    
    # ===== 测试5: 状态快照 =====
    print("\n[测试5] 状态快照")
    db.add_state(0, {
        'weights': np.array([0.3, 0.4, 0.3]),
        'gamma': 0.5,
        'hypervolume': 12345.6,
        'n_pareto': 3,
        'llm_focus_mu': np.array([4.5, 20.0, 2.5, 4.0]),
        'llm_focus_sigma': np.array([0.3, 5.0, 0.5, 0.05])
    })
    
    hv_hist = db.get_hv_history()
    assert len(hv_hist) == 1
    assert hv_hist[0] == 12345.6
    print(f"  ✓ 状态快照正确，HV={hv_hist[0]:.1f}")
    
    # ===== 测试6: Hypervolume计算 =====
    print("\n[测试6] Hypervolume计算")
    
    # 2D测试
    pf_2d = np.array([[1.0, 3.0], [2.0, 1.0]])
    ref_2d = np.array([4.0, 4.0])
    hv_2d = _hv_2d(pf_2d, ref_2d)
    # 面积 = (4-2)*(4-1) + (2-1)*(4-3) = 6 + 1 = 7
    # 实际：点(2,1): width=4-2=2, height=4-1=3 → 6; 点(1,3): width=2-1=1, height=4-3=1 → 1
    print(f"  2D HV = {hv_2d:.1f} (期望 7.0)")
    assert abs(hv_2d - 7.0) < 0.1, f"2D HV错误: {hv_2d}"
    
    # 3D测试
    pf_3d = np.array([
        [1.0, 3.0, 2.0],
        [2.0, 1.0, 3.0],
        [3.0, 2.0, 1.0]
    ])
    ref_3d = np.array([4.0, 4.0, 4.0])
    hv_3d = compute_hypervolume(pf_3d, ref_3d)
    print(f"  3D HV = {hv_3d:.1f}")
    assert hv_3d > 0, "3D HV应 > 0"
    
    # 空输入
    hv_empty = compute_hypervolume(np.empty((0, 3)), ref_3d)
    assert hv_empty == 0.0
    print(f"  空输入 HV = {hv_empty}")
    
    # ===== 测试7: 统计 =====
    print("\n[测试7] 统计信息")
    stats = db.get_statistics()
    print(f"  {stats}")
    assert stats['n_total'] == 10
    assert stats['n_valid'] == 9
    
    db.close()
    db2.close()
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)