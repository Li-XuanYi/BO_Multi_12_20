# LLMBO-MO 技术详解文档

本文档详细解释 LLMBO-MO 算法的完整流程，包括 WarmStart 初始化模块、Region-Lifted GP 模块，以及代码位置和关键实现细节。

---

## 目录

1. [整体架构](#一整体架构)
2. [WarmStart 模块详解](#二warmstart-模块详解)
3. [Region-Lifted GP 模块详解](#三region-lifted-gp-模块详解)
4. [Prompt 系统](#四prompt-系统)
5. [Pydantic 配置系统](#五pydantic-配置系统)
6. [关键代码位置汇总](#六关键代码位置汇总)

---

## 一、整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        WarmStart 执行流程                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 调用入口 (optimizer.py:808)                                             │
│     └── llm.generate_warmstart_candidates(n=10)                             │
│                                                                             │
│  2. 检查磁盘缓存 (llm_interface.py:1433-1459)                               │
│     ├── cache_hit=True  -> 直接返回缓存的候选点                               │
│     └── cache_hit=False -> 继续生成                                          │
│                                                                             │
│  3. 多批次 LLM 调用 (llm_interface.py:1461-1499)                            │
│     ├── 构建 Prompt (warmstart_prompt.py:264-338)                           │
│     │   └── 选择模板: basic/problem/detailed                                │
│     ├── 调用 LLM API (llm_interface.py:282-355)                             │
│     │   └── OpenAI/Anthropic/Mock 后端                                      │
│     └── 解析响应 (llm_interface.py:637-664)                                 │
│         └── validate_candidate() -> 边界检查 + dSOC约束检查                    │
│                                                                             │
│  4. 候选不足时补充 (llm_interface.py:1501-1505)                             │
│     └── physics_informed_warmstart()                                        │
│         └── 15个预定义策略点 + LHS采样                                       │
│                                                                             │
│  5. Portfolio 选择 (llm_interface.py:1507-1536)                             │
│     ├── 包装为 WarmStartCandidate                                           │
│     ├── 调用 select_warmstart_portfolio()                                   │
│     │   ├── filter_warmstart_candidates() -> 过滤无效点                      │
│     │   └── 迭代贪心选择:                                                   │
│     │       ├── 质量评分: confidence - soft_penalty + monotone_bonus        │
│     │       ├── 多样性评分: min_dist_to_selected                            │
│     │       └── 综合评分: quality + 0.45*diversity                          │
│     └── 不足时再次补充 fallback 点                                          │
│                                                                             │
│  6. 保存缓存 (llm_interface.py:1554-1559)                                   │
│     └── 保存候选池和选择结果到磁盘                                           │
│                                                                             │
│  7. 返回最终候选点 (llm_interface.py:1562)                                  │
│     └── n_warmstart (默认10个) 初始化点                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、入口调用

### 2.1 主优化器调用位置

**文件**: `llmbo/optimizer.py`  
**函数**: `run_initialization()`  
**行号**: 788-827

```python
def run_initialization(self) -> None:
    n_warmstart, n_random_init = self._resolve_init_counts()
    logger.info("=" * 60)
    logger.info(
        "Initialization: strategy=%s warmstart=%d random_init=%d",
        self.cfg.get("init_strategy", "manual"),
        n_warmstart,
        n_random_init,
    )
    logger.info("=" * 60)

    scheduled: List[Tuple[str, np.ndarray]] = []
    
    if n_warmstart > 0:
        warmstart_points = self.llm.generate_warmstart_candidates(
            n=n_warmstart,
            batch_size=int(self.cfg["warmstart_batch_size"]),
            max_attempts=int(self.cfg["warmstart_max_attempts"]),
        )
        if hasattr(self.llm, "get_warmstart_summary"):
            self._warmstart_portfolio_summary = self.llm.get_warmstart_summary()
        scheduled.extend(("llm_warmstart", theta) for theta in warmstart_points)

    if n_random_init > 0:
        random_points = self._get_random_init_points(n_random_init, seed=...)
        scheduled.extend(("random_init", theta) for theta in random_points)
```

**流程说明**:
1. 解析初始化计数 (`n_warmstart`, `n_random_init`)
2. 调用 `generate_warmstart_candidates()` 生成 WarmStart 点
3. 调用 `_get_random_init_points()` 生成随机初始化点
4. 去重后依次评估每个点

---

## 三、WarmStart 主生成函数

### 3.1 主入口

**文件**: `llm/llm_interface.py`  
**函数**: `generate_warmstart_candidates()`  
**行号**: 1408-1562

**完整函数签名**:
```python
def generate_warmstart_candidates(
    self,
    n: int = 15,                    # 需要返回的候选点数量
    batch_size: int = 20,           # 每批请求 LLM 的候选数
    max_attempts: Optional[int] = None,  # 最大尝试批次
) -> List[np.ndarray]:
```

**执行步骤**:

#### Step 1: 检查磁盘缓存 (行 1433-1459)
```python
disk_cache = self._load_warmstart_disk_cache()
cache_hit = False
if disk_cache is not None:
    cached_selected = self._coerce_theta_list(disk_cache.get("final_selected"))
    cached_pool = self._coerce_theta_list(disk_cache.get("candidate_pool"))
    if self._warmstart_cache_use_selected and len(cached_selected) >= int(n):
        # 使用缓存的已选择点
        return candidates
    if cached_pool:
        all_candidates = [self._parser.repair_theta(theta) for theta in cached_pool]
        cache_hit = True
```

#### Step 2: 多批次 LLM 调用 (行 1461-1499)
```python
for batch_idx in range(max_attempts):
    if cache_hit:
        break
    if len(all_candidates) >= target_pool:
        break

    request_size = max(int(batch_size), min(target_pool - len(all_candidates), target_pool))
    prompt = self._render_warmstart_prompt(request_size)

    batch: List[np.ndarray] = []
    for retry_idx in range(self._warmstart_max_retries + 1):
        responses = self._caller.call(
            prompt,
            temperature=self._warmstart_temperature,
            max_tokens=self._warmstart_max_tokens,
        )
        batch = self._parser.parse_candidates(responses)
        if batch:
            break
```

#### Step 3: 物理启发式补充 (行 1501-1505)
```python
if len(all_candidates) < target_pool:
    shortage = target_pool - len(all_candidates)
    logger.info("  LLM 候选不足，补充 %d 个物理启发式候选点", shortage)
    all_candidates.extend(self._fallback.physics_informed_warmstart(shortage))
```

#### Step 4: Portfolio 选择 (行 1507-1536)
```python
if self._enable_warmstart_portfolio:
    wrapped = [
        WarmStartCandidate(theta=np.asarray(theta, dtype=float), source="llm_pool", raw_index=i)
        for i, theta in enumerate(all_candidates)
    ]
    cfg = WarmStartSelectionConfig(
        n_select=int(n),
        bounds=self._bounds,
        hard_dsoc_sum_max=float(self._dsoc_sum_max),
        soft_dsoc_sum_max=float(self._safe_dsoc_sum_max or self._dsoc_sum_max),
        diversity_weight=float(self._warmstart_diversity_weight),
        soft_penalty_weight=float(self._warmstart_soft_penalty_weight),
        monotone_bonus=float(self._warmstart_monotone_bonus),
        archive_bonus_weight=float(self._warmstart_archive_bonus_weight),
        boundary_probe_limit=int(self._warmstart_boundary_probe_limit),
    )
    selected, summary = select_warmstart_portfolio(wrapped, cfg)
    candidates = [np.asarray(item.theta, dtype=float).copy() for item in selected]
```

#### Step 5: 保存缓存 (行 1554-1559)
```python
self._save_warmstart_disk_cache(
    candidate_pool=all_candidates,
    selected=candidates,
    summary=self._warmstart_summary,
    target_pool=target_pool,
)
```

---

## 四、Prompt 构建系统

### 4.1 Prompt 构建器

**文件**: `llm/warmstart_prompt.py`  
**类**: `WarmStartPromptContextBuilder`  
**行号**: 238-361

```python
class WarmStartPromptContextBuilder:
    """Build placeholder values for the warm-start prompt templates."""

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        battery_name: Optional[str],
        param_set: str,
        soc_start: float,
        soc_end: float,
        dsoc_sum_max: float = DEFAULT_DSOC_SUM_MAX,
        safe_dsoc_sum_max: Optional[float] = None,
        few_shot_examples: Optional[Sequence[Mapping[str, object]]] = None,
    ):
```

**核心方法** `build()` (行 264-338):
```python
def build(self, num_recommendation: int) -> Dict[str, str]:
    """构建 Prompt 模板所需的占位符值"""
    return {
        "NUM_RECOMMENDATION": str(int(num_recommendation)),
        "BATTERY_NAME": battery_name,
        "PARAM_SET_DISPLAY": param_set_display,
        "SOC_START": self._format_soc(self._soc_start),
        "SOC_END": self._format_soc(self._soc_end),
        "I1_RANGE": self._format_range("I1", unit="A"),
        "I2_RANGE": self._format_range("I2", unit="A"),
        "I3_RANGE": self._format_range("I3", unit="A"),
        "DSOC1_RANGE": self._format_range("dSOC1"),
        "DSOC2_RANGE": self._format_range("dSOC2"),
        "DSOC_SUM_MAX": f"{self._dsoc_sum_max:.2f}",
        "SAFE_DSOC_SUM_MAX": f"{self._safe_dsoc_sum_max:.2f}",
        "TASK_BRIEF": task_brief,
        "OBJECTIVE_SUMMARY": objective_summary,
        "COLLECTION_OBJECTIVE": collection_objective,
        "PROBLEM_DETAIL": problem_detail,
        "EXPERT_KNOWLEDGE": expert_knowledge,
        "TRADEOFF_BUCKETS": tradeoff_buckets,
        "ANTI_COLLAPSE_RULES": anti_collapse_rules,
        "ANTI_PATTERNS": anti_patterns,
        "FEW_SHOT_BLOCK": few_shot_examples,
        "NEGATIVE_EXAMPLE_BLOCK": negative_examples,
        "OUTPUT_SCHEMA": '[{"I1": value, "I2": value, "I3": value, "dSOC1": value, "dSOC2": value}, ...]',
    }
```

### 4.2 模板渲染

**文件**: `llm/warmstart_prompt.py`  
**函数**: `render_warmstart_prompt()`  
**行号**: 364-381

```python
def render_warmstart_prompt(
    level: str,                    # "none", "partial", "full"
    context: Mapping[str, str],    # 占位符值
    template_dir: Optional[Path] = None,
) -> str:
    """渲染 WarmStart Prompt 模板"""
    WARMSTART_TEMPLATE_MAP = {
        "none": "basic",      # 基础模板
        "partial": "problem", # 问题描述模板
        "full": "detailed",   # 详细模板
    }
    template_name = WARMSTART_TEMPLATE_MAP[level]
    renderer = WarmStartTemplateRenderer(template_dir=template_dir)
    prompt = renderer.render(template_name, context)
    return prompt
```

### 4.3 模板文件

**位置**: `llm/templates/warmstart/`

| 模板文件 | 级别 | 用途 |
|---------|------|------|
| `basic.txt` | none | 最小化信息，仅包含基本约束 |
| `problem.txt` | partial | 包含问题描述和权衡桶 |
| `detailed.txt` | full | 完整信息，包含专家知识、Few-shot示例、反模式 |

### 4.4 电池元数据注册表

**文件**: `llm/warmstart_prompt.py`  
**行号**: 51-65

```python
BATTERY_METADATA_REGISTRY: Dict[str, BatteryPromptMetadata] = {
    "Chen2020": BatteryPromptMetadata(
        param_set="Chen2020",
        battery_name="LG INR21700-M50",
        chemistry="NMC811/Graphite",
        nominal_capacity_ah=5.0,
        param_set_display="Chen2020 parameter set",
        expert_knowledge=(
            "Increasing I1 and I2 usually shortens charging time but raises peak temperature and aging risk.",
            "A larger dSOC1 keeps the cell at high current for longer, which is usually fast but thermally aggressive.",
            ...
        ),
    ),
}
```

### 4.5 权衡桶定义

**文件**: `llm/warmstart_prompt.py`  
**行号**: 67-74

```python
DEFAULT_TRADEOFF_BUCKETS: Tuple[Tuple[str, str], ...] = (
    ("fast_charge", "time-first; use stronger early current while keeping the late stage controlled"),
    ("thermal_safe", "temperature-first; prefer cooler interior points and lower late-stage stress"),
    ("aging_safe", "aging-first; taper meaningfully into the high-SOC region"),
    ("balanced", "balanced trade-off; avoid extreme current or SOC-span choices"),
    ("front_loaded_fast", "aggressive early acceleration followed by a clearly safer tail"),
    ("high_margin_safe", "leave obvious dSOC safety margin and stay well inside the feasible region"),
)
```

---

## 五、Portfolio 选择器

### 5.1 主选择函数

**文件**: `llmbo/warmstart_selector.py`  
**函数**: `select_warmstart_portfolio()`  
**行号**: 176-281

```python
def select_warmstart_portfolio(
    candidates: Sequence[WarmStartCandidate | np.ndarray],
    cfg: WarmStartSelectionConfig,
    *,
    archive_points: Optional[np.ndarray] = None,
) -> Tuple[List[WarmStartCandidate], Dict[str, Any]]:
    """Select a small warm-start portfolio from an over-generated pool."""
```

**算法流程**:
```python
# 1. 过滤无效候选
valid, summary = filter_warmstart_candidates(candidates, cfg)

# 2. 迭代贪心选择
while remaining and len(selected) < n_select:
    best_idx: Optional[int] = None
    best_score = -float("inf")
    
    for idx, candidate in enumerate(remaining):
        # 检查边界探索限制
        is_boundary = _is_boundary_probe(candidate, cfg.soft_dsoc_sum_max, cfg.boundary_probe_margin)
        if is_boundary and boundary_selected >= cfg.boundary_probe_limit:
            continue
        
        # 计算质量分
        quality = _candidate_quality(
            candidate,
            soft_limit=cfg.soft_dsoc_sum_max,
            hard_limit=cfg.hard_dsoc_sum_max,
            soft_penalty_weight=cfg.soft_penalty_weight,
            monotone_bonus=cfg.monotone_bonus,
        )
        
        # 计算多样性奖励
        diversity = 0.0
        if selected:
            x = _normalized(candidate.theta, lo, hi)
            selected_n = np.vstack([_normalized(item.theta, lo, hi) for item in selected])
            diversity = float(np.min(np.linalg.norm(selected_n - x[None, :], axis=1)))
        
        # 计算档案奖励
        archive = _archive_bonus(candidate.theta, archive_points, lo, hi)
        
        # 综合评分
        score = (
            quality
            + float(cfg.diversity_weight) * diversity
            + float(cfg.archive_bonus_weight) * archive
        )
        
        if score > best_score:
            best_score = score
            best_idx = idx
    
    # 选择最佳候选
    chosen = remaining.pop(best_idx)
    selected.append(chosen)
```

### 5.2 候选数据结构

**文件**: `llmbo/warmstart_selector.py`  
**行号**: 25-44

```python
@dataclasses.dataclass
class WarmStartCandidate:
    theta: np.ndarray                    # 5D参数 [I1, I2, I3, dSOC1, dSOC2]
    source: str = "llm"                  # 来源: llm, fallback, physics
    confidence: float = 0.5              # LLM置信度
    style: str = "unknown"               # 风格标签
    risk_flags: Tuple[str, ...] = ()     # 风险标记
    rationale: str = ""                  # 选择理由
    raw_index: int = 0                   # 原始索引

@dataclasses.dataclass
class WarmStartSelectionConfig:
    n_select: int                        # 需要选择的数量
    bounds: Dict[str, Tuple]             # 参数边界
    hard_dsoc_sum_max: float = 0.70      # 硬约束
    soft_dsoc_sum_max: float = 0.65      # 软约束
    diversity_weight: float = 0.45       # 多样性权重
    soft_penalty_weight: float = 0.65    # 软约束惩罚权重
    monotone_bonus: float = 0.08         # 单调性奖励
    archive_bonus_weight: float = 0.0    # 历史档案奖励
    boundary_probe_limit: int = 1        # 边界探索点限制
    dedup_decimals: int = 4              # 去重精度
```

### 5.3 质量评分函数

**文件**: `llmbo/warmstart_selector.py`  
**函数**: `_candidate_quality()`  
**行号**: 93-109

```python
def _candidate_quality(
    candidate: WarmStartCandidate,
    *,
    soft_limit: float,
    hard_limit: float,
    soft_penalty_weight: float,
    monotone_bonus: float,
) -> float:
    """计算候选点质量分数"""
    theta = np.asarray(candidate.theta, dtype=float).ravel()
    confidence = float(np.clip(candidate.confidence, 0.0, 1.0))
    dsoc_sum = float(theta[3] + theta[4])
    
    # 软约束越界惩罚
    denom = max(hard_limit - soft_limit, 1e-12)
    soft_over = max(0.0, dsoc_sum - soft_limit) / denom
    
    # 基础分数: 置信度 - 软约束惩罚
    score = confidence - soft_penalty_weight * soft_over
    
    # 单调性奖励 (I1 >= I2 >= I3)
    if _is_monotone(theta):
        score += float(monotone_bonus)
    
    return float(score)
```

### 5.4 候选过滤函数

**文件**: `llmbo/warmstart_selector.py`  
**函数**: `filter_warmstart_candidates()`  
**行号**: 126-173

```python
def filter_warmstart_candidates(
    candidates: Iterable[WarmStartCandidate | np.ndarray],
    cfg: WarmStartSelectionConfig,
) -> Tuple[List[WarmStartCandidate], Dict[str, Any]]:
    """过滤无效候选点"""
    filtered = {
        "non_finite": 0,    # 非有限值
        "shape": 0,         # 维度错误
        "bounds": 0,        # 超出边界
        "hard_dsoc": 0,     # 违反硬约束
        "duplicate": 0,     # 重复点
    }
    
    for raw_index, item in enumerate(candidates):
        candidate = _as_candidate(item, raw_index=raw_index)
        theta = np.asarray(candidate.theta, dtype=float).ravel()
        
        # 检查维度
        if theta.size != len(PARAM_NAMES):
            filtered["shape"] += 1
            continue
        
        # 检查有限值
        if not np.all(np.isfinite(theta)):
            filtered["non_finite"] += 1
            continue
        
        # 检查边界
        if np.any(theta < lo - 1e-12) or np.any(theta > hi + 1e-12):
            filtered["bounds"] += 1
            continue
        
        # 检查硬约束 dSOC1 + dSOC2 <= 0.70
        if dsoc_sum_violates_limit(theta[3], theta[4], dsoc_sum_max=cfg.hard_dsoc_sum_max):
            filtered["hard_dsoc"] += 1
            continue
        
        # 去重检查
        key = tuple(np.round(theta, cfg.dedup_decimals))
        if key in seen:
            filtered["duplicate"] += 1
            continue
        
        seen.add(key)
        valid.append(dataclasses.replace(candidate, theta=theta))
    
    return valid, summary
```

---

## 六、物理启发式回退

### 6.1 回退类定义

**文件**: `llm/llm_interface.py`  
**类**: `PhysicsHeuristicFallback`  
**行号**: 670-767

```python
class PhysicsHeuristicFallback:
    """LLM 不可用或响应无效时的回退采样策略"""

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        dsoc_sum_max: float = _DSOC_SUM_MAX,
        soft_dsoc_sum_max: Optional[float] = LLM_SAFE_DSOC_SUM_MAX,
    ):
        self._lo = np.array([param_bounds[k][0] for k in PARAM_KEYS])
        self._hi = np.array([param_bounds[k][1] for k in PARAM_KEYS])
        self._dsoc_sum_max = float(dsoc_sum_max)
        self._soft_dsoc_sum_max = ...
```

### 6.2 预定义策略点

**文件**: `llm/llm_interface.py`  
**函数**: `physics_informed_warmstart()`  
**行号**: 697-749

```python
def physics_informed_warmstart(self, n: int) -> List[np.ndarray]:
    """
    基于领域知识的先验候选点（覆盖 Pareto 极端方向）
    超出 n=15 的部分由 LHS 补全
    """
    # 格式：[I1, I2, I3, dSOC1, dSOC2]
    prior_points = [
        # 激进快充：高电流，小 SOC 区间
        np.array([5.5, 4.5, 2.8, 0.20, 0.20]),
        # 保守安全：低电流，大 SOC 区间
        np.array([2.5, 2.5, 2.0, 0.35, 0.25]),
        # 均衡折衷
        np.array([4.0, 3.5, 2.5, 0.25, 0.20]),
        # 偏快，温度控制（I3 低）
        np.array([5.0, 4.0, 2.2, 0.20, 0.25]),
        # 低老化（I2/I3 低，高 SOC 区间小电流）
        np.array([3.5, 3.0, 2.0, 0.30, 0.28]),
        # 大 I1 快速启动，后段保守
        np.array([5.8, 3.0, 2.0, 0.18, 0.22]),
        # 平衡温度和老化
        np.array([3.0, 2.8, 2.2, 0.38, 0.28]),
    ]

    # 极端方向点（覆盖更多 Pareto 区域）
    extreme_points = [
        # 极端时间优先：最大电流，最小 SOC 区间
        np.array([6.0, 5.0, 3.0, 0.15, 0.15]),
        # 极端温度优先：最小电流，最大 SOC 区间
        np.array([2.0, 2.0, 2.0, 0.40, 0.30]),
        # 极端老化优先：渐进电流，大最终 SOC 区间
        np.array([3.5, 3.0, 2.5, 0.35, 0.30]),
        # 时间-温度权衡：高 I1，低 I2/I3
        np.array([5.8, 3.5, 2.2, 0.18, 0.22]),
        # 时间-老化权衡：大 I1，小 I3，大 dSOC2
        np.array([5.5, 4.0, 2.0, 0.20, 0.35]),
        # 温度-老化权衡：低电流，大 SOC 区间
        np.array([2.8, 2.5, 2.2, 0.38, 0.30]),
        # 均衡策略 2
        np.array([4.2, 3.8, 2.6, 0.22, 0.24]),
        # 均衡策略 3
        np.array([3.8, 3.2, 2.4, 0.28, 0.26]),
    ]

    all_prior = prior_points + extreme_points
    candidates = [self._repair_theta(p) for p in all_prior[:min(n, len(all_prior))]]

    if len(candidates) < n:
        candidates.extend(self.lhs_candidates(n - len(candidates), seed=42))

    return candidates[:n]
```

**预定义策略点汇总**:

| 类型 | 点 [I1, I2, I3, dSOC1, dSOC2] | 特点 |
|------|-------------------------------|------|
| 基础点 | [5.5, 4.5, 2.8, 0.20, 0.20] | 激进快充 |
| 基础点 | [2.5, 2.5, 2.0, 0.35, 0.25] | 保守安全 |
| 基础点 | [4.0, 3.5, 2.5, 0.25, 0.20] | 均衡折衷 |
| 基础点 | [5.0, 4.0, 2.2, 0.20, 0.25] | 偏快+温度控制 |
| 基础点 | [3.5, 3.0, 2.0, 0.30, 0.28] | 低老化优先 |
| 基础点 | [5.8, 3.0, 2.0, 0.18, 0.22] | 快速启动+保守后段 |
| 基础点 | [3.0, 2.8, 2.2, 0.38, 0.28] | 温度老化平衡 |
| 极端点 | [6.0, 5.0, 3.0, 0.15, 0.15] | 极端时间优先 |
| 极端点 | [2.0, 2.0, 2.0, 0.40, 0.30] | 极端温度优先 |
| 极端点 | [3.5, 3.0, 2.5, 0.35, 0.30] | 极端老化优先 |
| 极端点 | [5.8, 3.5, 2.2, 0.18, 0.22] | 时间-温度权衡 |
| 极端点 | [5.5, 4.0, 2.0, 0.20, 0.35] | 时间-老化权衡 |
| 极端点 | [2.8, 2.5, 2.2, 0.38, 0.30] | 温度-老化权衡 |
| 极端点 | [4.2, 3.8, 2.6, 0.22, 0.24] | 均衡策略 2 |
| 极端点 | [3.8, 3.2, 2.4, 0.28, 0.26] | 均衡策略 3 |

### 6.3 LHS 采样

**文件**: `llm/llm_interface.py`  
**函数**: `lhs_candidates()`  
**行号**: 751-767

```python
def lhs_candidates(self, n: int, seed: int = 0) -> List[np.ndarray]:
    """Latin Hypercube Sampling，生成边界内均匀分布候选点"""
    if n <= 0:
        return []
    rng = np.random.default_rng(seed)
    d = len(PARAM_KEYS)
    samples = np.zeros((n, d))
    
    # 生成 LHS 样本
    for j in range(d):
        perm = rng.permutation(n)
        samples[:, j] = (perm + rng.random(n)) / n
    
    # 映射到参数边界
    candidates = []
    for i in range(n):
        theta = self._lo + samples[i] * (self._hi - self._lo)
        candidates.append(self._repair_theta(theta))
    
    return candidates
```

---

## 七、响应解析与验证

### 7.1 解析器类

**文件**: `llm/llm_interface.py`  
**类**: `ResponseParser`  
**行号**: 414-664

```python
class ResponseParser:
    """
    解析 LLM 响应，提取并验证 5D 候选点
    验证规则：
      1. 每个参数在各自边界内
      2. dSOC1 + dSOC2 <= 0.70（防止 dSOC3 <= 0）
    """

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        dsoc_sum_max: float = _DSOC_SUM_MAX,
        soft_dsoc_sum_max: Optional[float] = LLM_SAFE_DSOC_SUM_MAX,
    ):
        self._bounds = param_bounds
        self._dsoc_sum_max = float(dsoc_sum_max)
        self._soft_dsoc_sum_max = ...
```

### 7.2 JSON 提取

**文件**: `llm/llm_interface.py`  
**函数**: `extract_json()`  
**行号**: 436-465

```python
@staticmethod
def extract_json(text: str) -> Optional[Any]:
    """从 LLM 响应文本中提取 JSON，容错处理"""
    if not text or not text.strip():
        return None
    text = text.strip()

    # 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # markdown 代码块
    m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 提取第一个 JSON 数组或对象
    for pattern in [r'(\[[\s\S]*\])', r'(\{[\s\S]*\})']:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    return None
```

### 7.3 候选验证

**文件**: `llm/llm_interface.py`  
**函数**: `validate_candidate()`  
**行号**: 467-489

```python
def validate_candidate(self, d: Dict) -> Optional[np.ndarray]:
    """验证单个候选字典，返回 5D ndarray 或 None"""
    try:
        values = []
        for key in PARAM_KEYS:
            val = float(d[key])
            lo, hi = self._bounds[key]
            if val < lo or val > hi:
                logger.debug("候选点 %s=%.4f 越界 [%.2f, %.2f]", key, val, lo, hi)
                return None
            values.append(val)

        # 额外检查 dSOC 约束
        dSOC_sum = values[3] + values[4]  # dSOC1 + dSOC2
        if dsoc_sum_violates_limit(values[3], values[4], dsoc_sum_max=self._dsoc_sum_max):
            logger.debug("dSOC1+dSOC2=%.3f > %.2f，候选无效", dSOC_sum, self._dsoc_sum_max)
            return None

        return self.repair_theta(np.array(values, dtype=float))

    except (KeyError, TypeError, ValueError) as e:
        logger.debug("候选点验证失败: %s", e)
        return None
```

### 7.4 约束修复

**文件**: `llm/llm_interface.py`  
**函数**: `repair_theta()`  
**行号**: 491-505

```python
def repair_theta(self, theta: np.ndarray) -> np.ndarray:
    """修复候选点使其满足约束"""
    x = np.asarray(theta, dtype=float).ravel().copy()
    if x.size != len(PARAM_KEYS):
        raise ValueError(f"Expected {len(PARAM_KEYS)} parameters, got {x.size}")

    # 裁剪到边界
    for idx, key in enumerate(PARAM_KEYS):
        lo, hi = self._bounds[key]
        x[idx] = float(np.clip(x[idx], lo, hi))

    # 修复 dSOC 约束
    repair_limit = self._soft_dsoc_sum_max or self._dsoc_sum_max
    if dsoc_sum_violates_limit(x[3], x[4], dsoc_sum_max=repair_limit):
        x[3], x[4] = project_dsoc_pair(x[3], x[4], dsoc_sum_max=repair_limit)
        x[3] = float(np.clip(x[3], self._bounds["dSOC1"][0], self._bounds["dSOC1"][1]))
        x[4] = float(np.clip(x[4], self._bounds["dSOC2"][0], self._bounds["dSOC2"][1]))
    return x
```

### 7.5 批量解析

**文件**: `llm/llm_interface.py`  
**函数**: `parse_candidates()`  
**行号**: 637-664

```python
def parse_candidates(self, responses: List[str]) -> List[np.ndarray]:
    """从多个 LLM 响应中解析并合并所有有效候选点（已去重）"""
    all_valid: List[np.ndarray] = []
    seen = set()

    for resp_idx, text in enumerate(responses):
        parsed = self.extract_json(text)
        if parsed is None:
            continue

        candidates = [parsed] if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])

        cnt = 0
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            theta = self.validate_candidate(cand)
            if theta is not None:
                h = tuple(theta.round(4).tolist())
                if h not in seen:
                    seen.add(h)
                    all_valid.append(theta)
                    cnt += 1

        logger.debug("响应 %d: 解析出 %d 个有效候选点", resp_idx, cnt)

    logger.info("ResponseParser: 共 %d 个有效候选点（%d 个响应）", len(all_valid), len(responses))
    return all_valid
```

---

## 八、缓存机制

### 8.1 加载缓存

**文件**: `llm/llm_interface.py`  
**函数**: `_load_warmstart_disk_cache()`  
**行号**: 1019-1034

```python
def _load_warmstart_disk_cache(self) -> Optional[Dict[str, Any]]:
    """从磁盘加载 warmstart 缓存"""
    if self._warmstart_cache_path is None:
        return None
    if self._warmstart_cache_mode not in {"read", "read_write"}:
        return None
    if not self._warmstart_cache_path.exists():
        return None
    try:
        with open(self._warmstart_cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read warmstart cache %s: %s", self._warmstart_cache_path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    return payload
```

### 8.2 保存缓存

**文件**: `llm/llm_interface.py`  
**函数**: `_save_warmstart_disk_cache()`  
**行号**: 1036-1063

```python
def _save_warmstart_disk_cache(
    self,
    *,
    candidate_pool: List[np.ndarray],
    selected: List[np.ndarray],
    summary: Dict[str, Any],
    target_pool: int,
) -> None:
    """保存 warmstart 结果到磁盘"""
    if self._warmstart_cache_path is None:
        return
    if self._warmstart_cache_mode not in {"write", "read_write"}:
        return
    
    payload = {
        "version": 1,
        "backend": str(self._config.backend),
        "model": str(self._config.model),
        "temperature": float(self._warmstart_temperature),
        "target_pool": int(target_pool),
        "candidate_pool": [np.asarray(theta, dtype=float).ravel().tolist() for theta in candidate_pool],
        "final_selected": [np.asarray(theta, dtype=float).ravel().tolist() for theta in selected],
        "summary": summary,
    }
    try:
        self._warmstart_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._warmstart_cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.warning("Failed to save warmstart cache %s: %s", self._warmstart_cache_path, exc)
```

---

## 九、配置参数

### 9.1 默认配置

**文件**: `llmbo/optimizer.py`  
**行号**: 69-92

```python
DEFAULT_CONFIG = {
    "n_warmstart": 10,                    # WarmStart 点数
    "n_random_init": 3,                   # 随机初始化点数
    "init_strategy": "manual",            # 初始化策略
    "init_budget": None,                  # 初始化预算
    "warmstart_ratio": 0.5,               # WarmStart 比例
    "fixed_init_points": None,            # 固定初始化点
    
    # LLM 调用配置
    "warmstart_batch_size": 10,           # 每批请求大小
    "warmstart_max_attempts": 4,          # 最大尝试批次
    "warmstart_max_retries": 3,           # 每批重试次数
    "warmstart_max_tokens": 2500,         # 最大 token 数
    "warmstart_temperature": None,        # 采样温度
    
    # Portfolio 选择器配置
    "enable_warmstart_portfolio": True,   # 启用 Portfolio 选择器
    "warmstart_pool_size": 16,            # 候选池目标大小
    "warmstart_diversity_weight": 0.45,   # 多样性权重
    "warmstart_soft_penalty_weight": 0.65,# 软约束惩罚
    "warmstart_monotone_bonus": 0.08,     # 单调性奖励
    "warmstart_archive_bonus_weight": 0.0,# 历史档案奖励
    "warmstart_boundary_probe_limit": 1,  # 边界探索点限制
    
    # 缓存配置
    "warmstart_cache_path": None,         # 缓存路径
    "warmstart_cache_mode": "read_write", # 缓存模式: read/write/read_write
    "warmstart_cache_use_selected": False,# 使用缓存的已选择点
}
```

### 9.2 配置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_warmstart` | 10 | 需要生成的 WarmStart 点数 |
| `n_random_init` | 3 | 随机初始化点数 |
| `warmstart_batch_size` | 10 | 每批请求 LLM 生成的候选数 |
| `warmstart_max_attempts` | 4 | 最大尝试批次 |
| `warmstart_pool_size` | 16 | 候选池目标大小（用于 Portfolio 选择） |
| `enable_warmstart_portfolio` | True | 是否启用 Portfolio 选择器 |
| `warmstart_diversity_weight` | 0.45 | 多样性评分权重 |
| `warmstart_soft_penalty_weight` | 0.65 | dSOC 软约束惩罚权重 |
| `warmstart_monotone_bonus` | 0.08 | 单调性奖励 (I1>=I2>=I3) |

---

## 十、关键代码位置汇总

| 功能 | 文件 | 行号 | 函数/类 |
|------|------|------|---------|
| 入口调用 | `optimizer.py` | 808 | `run_initialization()` |
| 主生成函数 | `llm_interface.py` | 1408 | `generate_warmstart_candidates()` |
| Prompt构建 | `warmstart_prompt.py` | 238 | `WarmStartPromptContextBuilder.build()` |
| 模板渲染 | `warmstart_prompt.py` | 364 | `render_warmstart_prompt()` |
| 模板文件 | `templates/warmstart/` | - | `basic.txt`, `problem.txt`, `detailed.txt` |
| Portfolio选择 | `warmstart_selector.py` | 176 | `select_warmstart_portfolio()` |
| 候选过滤 | `warmstart_selector.py` | 126 | `filter_warmstart_candidates()` |
| 质量评分 | `warmstart_selector.py` | 93 | `_candidate_quality()` |
| 物理回退 | `llm_interface.py` | 670 | `PhysicsHeuristicFallback` |
| 响应解析 | `llm_interface.py` | 414 | `ResponseParser` |
| 候选验证 | `llm_interface.py` | 467 | `validate_candidate()` |
| 约束修复 | `llm_interface.py` | 491 | `repair_theta()` |
| 缓存加载 | `llm_interface.py` | 1019 | `_load_warmstart_disk_cache()` |
| 缓存保存 | `llm_interface.py` | 1036 | `_save_warmstart_disk_cache()` |
| LHS采样 | `llm_interface.py` | 751 | `lhs_candidates()` |

---

## 十一、使用示例

### 11.1 基本使用

```python
from llm.llm_interface import build_llm_interface, DEFAULT_BOUNDS

# 构建 LLM 接口
llm = build_llm_interface(
    DEFAULT_BOUNDS,
    backend="openai",
    model="gpt-4.1-mini",
    enable_warmstart_portfolio=True,
    warmstart_pool_size=16,
)

# 生成 WarmStart 候选点
candidates = llm.generate_warmstart_candidates(n=10)
print(f"Generated {len(candidates)} warm-start points")

# 获取选择摘要
summary = llm.get_warmstart_summary()
print(summary)
```

### 11.2 使用缓存

```python
llm = build_llm_interface(
    DEFAULT_BOUNDS,
    backend="openai",
    model="gpt-4.1-mini",
    warmstart_cache_path="./cache/warmstart.json",
    warmstart_cache_mode="read_write",
)

# 如果缓存存在，将直接返回缓存结果
candidates = llm.generate_warmstart_candidates(n=10)
```

### 11.3 禁用 Portfolio 选择

```python
llm = build_llm_interface(
    DEFAULT_BOUNDS,
    backend="openai",
    enable_warmstart_portfolio=False,  # 禁用 Portfolio 选择
)

# 直接返回前 n 个候选点
candidates = llm.generate_warmstart_candidates(n=10)
```

---

## 十二、设计要点

1. **分层设计**: WarmStart 模块分为 LLM 生成、Portfolio 选择、物理回退三层
2. **容错机制**: LLM 失败时自动回退到物理启发式策略
3. **多样性保证**: Portfolio 选择器通过多样性评分确保候选点覆盖不同区域
4. **约束处理**: 硬约束（dSOC<=0.70）必须满足，软约束（dSOC<=0.65）影响评分
5. **可缓存**: 支持磁盘缓存避免重复调用 LLM API

---

## 二、WarmStart 模块详解

### 2.1 执行流程

```
WarmStart 执行流程

1. 调用入口 (optimizer.py:808)
   └── llm.generate_warmstart_candidates(n=10)

2. 检查磁盘缓存 (llm_interface.py:1433-1459)
   ├── cache_hit=True  -> 直接返回缓存的候选点
   └── cache_hit=False -> 继续生成

3. 多批次 LLM 调用 (llm_interface.py:1461-1499)
   ├── 构建 Prompt (warmstart_prompt.py:264-338)
   │   └── 选择模板: basic/problem/detailed/experimental
   ├── 调用 LLM API (llm_interface.py:282-355)
   │   └── OpenAI/Anthropic/Mock 后端
   └── 解析响应 (llm_interface.py:637-664)
       └── validate_candidate() -> 边界检查 + dSOC约束检查

4. 候选不足时补充 (llm_interface.py:1501-1505)
   └── physics_informed_warmstart()
       └── 15个预定义策略点 + LHS采样

5. Portfolio 选择 (llm_interface.py:1507-1536)
   ├── 包装为 WarmStartCandidate
   ├── 调用 select_warmstart_portfolio()
   │   ├── filter_warmstart_candidates() -> 过滤无效点
   │   └── 迭代贪心选择:
   │       ├── 质量评分: confidence - soft_penalty + monotone_bonus
   │       ├── 多样性评分: min_dist_to_selected
   │       └── 综合评分: quality + 0.45*diversity
   └── 不足时再次补充 fallback 点

6. 保存缓存 (llm_interface.py:1554-1559)
   └── 保存候选池和选择结果到磁盘

7. 返回最终候选点 (llm_interface.py:1562)
   └── n_warmstart (默认10个) 初始化点
```

### 2.2 Prompt 模板级别

**文件**: `llm/warmstart_prompt.py`
**函数**: `render_warmstart_prompt()`

模板级别映射 (WARMSTART_TEMPLATE_MAP):

| level 参数 | 模板文件 | 说明 |
|-----------|---------|------|
| "none" | basic.txt | 最小化信息，仅包含基本约束 |
| "partial" | problem.txt | 包含问题描述和权衡桶 |
| "full" | detailed.txt | 完整信息，包含专家知识、Few-shot示例 |
| "experimental" | experimental.txt | 实验性模板，包含增强的物理引导 |

新增的 experimental 级别用于实验性研究，提供更丰富的物理引导和领域知识。

---

## 三、Region-Lifted GP 模块详解

### 3.1 核心概念

Region-Lifted GP 是 LLM 触点的第二个关键组件，在每个 BO 迭代中：

1. **查询 LLM**: 询问 "哪里可能有更好的充电协议？"
2. **获得建议**: LLM 返回一个区域 (region) 或点 (point)
3. **提升 GP**: 在区域内"抬高"高斯过程的均值预测
4. **指导搜索**: 采集函数更倾向于选择该区域内的点

### 3.2 两种工作模式

#### 模式 1: Standard Region-Lift

**文件**: `llmbo/region_lifted_gp.py`
**函数**: `evaluate_region_lift_on_pool()`

通过 Mean Shift 调整 GP 预测：
```
mean_lifted(x) = mean_gp(x) - shift(x)
shift(x) = λ_t × reliability × max(correlation(x, center), 0)
```

#### 模式 2: LGBO (Latent Gaussian Bayesian Optimization)

**文件**: `llmbo/region_lifted_gp.py`
**函数**: `build_lgbo_region_lift()`
**类**: `LGBORegionLiftBuildResult`

LGBO 模式实现 Proposition 1，通过 Region 耦合矩阵增强 GP。

启用条件：
```python
def is_lgbo_region_lift_mode(cfg):
    return cfg.region_lift_mode == "lgbo_proposition1"
```

### 3.3 Region 随机化控制

**控制模式**: shape_randomized

用于在 Region 推荐中引入随机性，避免 LLM 推荐过于集中。

### 3.4 解析与验证

**文件**: `llmbo/region_lifted_gp.py`
**函数**: `parse_region_preference_payload()`

解析 LLM 返回的 JSON 响应，支持：
- Point 类型: kind="point"，包含具体坐标
- Region 类型: kind="region"，包含 lb/ub 边界
- None 类型: kind="none"，表示 LLM 无建议

---

## 四、Prompt 系统

### 4.1 WarmStart Prompt

**文件**: `llm/warmstart_prompt.py`
**类**: `WarmStartPromptContextBuilder`

构建 Prompt 模板所需的占位符值，包含电池信息、参数范围、专家知识等。

### 4.2 Region Preference Prompt

**文件**: `llm/region_prompt.py`
**函数**: `render_region_preference_prompt()`

生成 Region 推荐的 Prompt，包含新要求：**mechanistic_thinking**

**关键要求**:
- mechanistic_thinking: 1-2 句话的机理解释，非逐步推理
- 证据层次: PRIMARY 领域知识 > SECONDARY 历史数据
- 反崩溃: 不以过去观测为中心，除非有机制支持

### 4.3 Mechanistic Thinking 字段

Region Prompt 要求 LLM 输出包含 mechanistic_thinking 字段，用于解释推荐的物理机制。

---

## 五、Pydantic 配置系统

### 5.1 配置类层次

**文件**: `config/schema.py`

使用 Pydantic v2 进行类型安全的配置管理。

关键配置类：
- `LLMWarmStartConfig`: WarmStart 配置
- `RegionLiftConfig`: Region-Lifted GP 配置

### 5.2 验证器

支持字段级别验证器 (@field_validator) 和模型级别验证器 (@model_validator)。

### 5.3 配置加载

**文件**: `config/load.py`

支持从 JSON、环境变量、CLI 参数加载配置。

---

## 六、关键代码位置汇总

### 6.1 WarmStart 模块

| 功能 | 文件 | 函数/类 |
|------|------|---------|
| 入口调用 | llmbo/optimizer.py | run_initialization() |
| 主生成函数 | llm/llm_interface.py | generate_warmstart_candidates() |
| Prompt构建 | llm/warmstart_prompt.py | WarmStartPromptContextBuilder.build() |
| Portfolio选择 | llmbo/warmstart_selector.py | select_warmstart_portfolio() |

### 6.2 Region-Lifted GP 模块

| 功能 | 文件 | 函数/类 |
|------|------|---------|
| Region Prompt | llm/region_prompt.py | render_region_preference_prompt() |
| Region解析 | llmbo/region_lifted_gp.py | parse_region_preference_payload() |
| Region评估 | llmbo/region_lifted_gp.py | evaluate_region_lift_on_pool() |
| LGBO模式 | llmbo/region_lifted_gp.py | build_lgbo_region_lift() |

### 6.3 配置系统

| 功能 | 文件 | 函数/类 |
|------|------|---------|
| 根配置 | config/schema.py | Config |
| WarmStart配置 | config/schema.py | LLMWarmStartConfig |
| Region配置 | config/schema.py | RegionLiftConfig |

---

文档版本: v2.0 | 更新日期: 2026-05-19
