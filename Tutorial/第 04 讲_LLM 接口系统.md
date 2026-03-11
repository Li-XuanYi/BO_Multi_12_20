# 第 04 讲：LLM 接口系统详解

> **学习目标**：深入理解 LLM 与优化框架的集成机制，掌握 Prompt 工程和响应解析技术

---

## 1. LLM 在框架中的角色

### 1.1 三个 Touchpoint

LLM 在 LLMBO-MO 中通过三个接口参与优化循环：

```
Touchpoint 1a: 初始化阶段
  │
  ├─► 生成耦合矩阵 W_time, W_temp, W_aging
  │   └─► 注入 GP 模型
  │
  ▼
Touchpoint 1b: Warmstart 阶段
  │
  ├─► 生成 N_ws 个初始候选点
  │   └─► PyBaMM 评估 → 填充数据库
  │
  ▼
Touchpoint 2: 迭代生成（每次迭代调用）
  │
  ├─► 接收优化状态（μ, σ, θ_best, ...）
  │   └─► 生成 N 个新候选点
  │       └─► 采集函数筛选 → PyBaMM 评估
```

### 1.2 LLM 的价值

| Touchpoint | 作用 | 替代方案 |
|------------|------|----------|
| 1a | 注入物理耦合先验 | 单位矩阵/人工设计 |
| 1b | 提供高质量初始点 | 随机采样/拉丁超立方 |
| 2 | 基于状态智能生成 | 随机扰动/进化策略 |

---

## 2. LLM 接口架构

### 2.1 核心组件

```python
# llm_interface.py 核心类结构

LLMConfig
  │
  ├─► 配置管理（backend, model, api_base, ...）
  │
  ▼
TemplateEngine
  │
  ├─► Prompt 模板渲染
  │
  ▼
LLMCaller
  │
  ├─► 多后端 API 调用（Ollama/OpenAI/Anthropic）
  │
  ▼
ResponseParser
  │
  ├─► JSON 提取与验证
  │
  ▼
PhysicsHeuristicFallback
  │
  ├─► LLM 失败时的回退策略
  │
  ▼
LLMInterface (门面类)
  │
  ├─► 对外提供三个 Touchpoint 方法
```

### 2.2 初始化流程

```python
# llm_interface.py:257-309
def build_llm_interface(
    param_bounds: Dict[str, Tuple[float, float]],
    backend: str = "openai",
    model: str = "gpt-4o",
    api_base: str = "https://api.openai.com/v1",
    api_key: str = None,
    n_samples: int = 5,
    temperature: float = 0.7,
    battery_model: str = "LG M50 (Chen2020)",
) -> LLMInterface:
    """工厂函数：构建 LLM 接口"""

    # Step 1: 创建配置
    config = LLMConfig(
        backend=backend,
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        max_tokens=2000,
    )

    # Step 2: 创建调用器
    caller = LLMCaller(config)

    # Step 3: 创建模板引擎
    template_engine = TemplateEngine(
        template_dir="templates",  # 模板目录
    )

    # Step 4: 创建解析器
    parser = ResponseParser(param_bounds)

    # Step 5: 创建回退策略
    fallback = PhysicsHeuristicFallback(param_bounds)

    # Step 6: 创建门面类
    return LLMInterface(
        caller=caller,
        template_engine=template_engine,
        parser=parser,
        fallback=fallback,
        param_bounds=param_bounds,
        battery_model=battery_model,
    )
```

---

## 3. 模板引擎详解

### 3.1 模板文件结构

```
templates/
│
├─ coupling_matrix.txt        # Touchpoint 1a
├─ warmstart_candidates.txt   # Touchpoint 1b
└─ iterative_candidates.txt   # Touchpoint 2
```

### 3.2 模板语法

**占位符格式**：`[PLACEHOLDER_NAME]`

**示例** (`templates/coupling_matrix.txt`)：

```
You are an expert in lithium-ion battery fast charging optimization.

We are optimizing a two-stage constant-current (CC-CC) charging protocol with three decision variables:
  - I₁ (Phase-1 current): [I1_RANGE] A
  - SOC₁ (switching SOC): [SOC1_RANGE]
  - I₂ (Phase-2 current): [I2_RANGE] A

The battery is [BATTERY_MODEL] with nominal capacity [Q_NOM] Ah, charged from SOC₀=[SOC0] to SOC_end=[SOC_END].

...（更多物理说明）

Respond with ONLY a JSON object containing three matrices, no other text:
{"W_time": [[w11,w12,w13],[w21,w22,w23],[w31,w32,w33]], ...}
```

### 3.3 模板渲染

```python
# llm_interface.py:83-125
class TemplateEngine:
    def __init__(self, template_dir: str = "templates"):
        self.template_dir = Path(template_dir)
        self._cache = {}  # 模板缓存

    def render(self, template_name: str, **kwargs) -> str:
        """渲染模板，替换占位符"""

        # 加载模板（带缓存）
        if template_name not in self._cache:
            template_path = self.template_dir / f"{template_name}.txt"
            with open(template_path, "r", encoding="utf-8") as f:
                self._cache[template_name] = f.read()

        template = self._cache[template_name]

        # 替换占位符
        result = template
        for key, value in kwargs.items():
            placeholder = f"[{key}]"
            result = result.replace(placeholder, str(value))

        return result
```

**使用示例**：

```python
engine = TemplateEngine()

prompt = engine.render(
    "coupling_matrix",
    I1_RANGE="3.0 - 7.0",
    SOC1_RANGE="0.10 - 0.70",
    I2_RANGE="1.0 - 5.0",
    BATTERY_MODEL="LG M50 (Chen2020)",
    Q_NOM="5.0",
    SOC0="0.035",
    SOC_END="0.80",
)
```

---

## 4. LLM 调用器

### 4.1 多后端支持

```python
# llm_interface.py:128-203
class LLMCaller:
    """
    多后端 LLM 调用器

    支持：
    - Ollama (本地部署)
    - OpenAI (GPT-4/3.5)
    - Anthropic (Claude)
    - Mock (测试)
    """

    def __init__(self, config: LLMConfig):
        self.config = config

        if config.backend == "ollama":
            self.api_base = config.api_base or "http://localhost:11434/api"
        elif config.backend == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=config.api_key, base_url=config.api_base)
        elif config.backend == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=config.api_key)
        elif config.backend == "mock":
            pass  # 测试模式
```

### 4.2 调用接口

```python
def call(self, prompt: str, system_prompt: str = None) -> str:
    """发送请求并返回响应文本"""

    if self.config.backend == "mock":
        return self._mock_call(prompt)

    elif self.config.backend == "ollama":
        return self._ollama_call(prompt)

    elif self.config.backend == "openai":
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content

    elif self.config.backend == "anthropic":
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system_prompt or "You are a helpful assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
```

### 4.3 重试与超时

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def call_with_retry(self, prompt: str, system_prompt: str = None) -> str:
    """带重试的调用（应对 API 临时故障）"""
    return self.call(prompt, system_prompt)
```

---

## 5. 响应解析器

### 5.1 JSON 提取

**挑战**：LLM 响应可能包含额外文本、markdown 格式等。

```python
# llm_interface.py:206-254
class ResponseParser:
    def __init__(self, param_bounds: Dict[str, Tuple[float, float]]):
        self.param_bounds = param_bounds

    def extract_json(self, text: str) -> Any:
        """从响应文本中提取 JSON"""

        # 尝试 1: 直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试 2: 查找第一个 { 或 [
        start_idx = -1
        for i, char in enumerate(text):
            if char in "{[":
                start_idx = i
                break

        if start_idx != -1:
            # 查找匹配的结束括号
            bracket = text[start_idx]
            end_bracket = "}" if bracket == "{" else "]"
            depth = 0
            for i in range(start_idx, len(text)):
                if text[i] == bracket:
                    depth += 1
                elif text[i] == end_bracket:
                    depth -= 1
                    if depth == 0:
                        json_str = text[start_idx:i+1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            pass

        # 尝试 3: 使用正则表达式
        json_pattern = r'(\{.*\}|\[.*\])'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        raise ValueError("无法从响应中提取 JSON")
```

### 5.2 边界验证

```python
def validate_candidates(self, candidates: List[Dict]) -> List[Dict]:
    """验证候选点是否在参数边界内"""

    valid = []
    for cand in candidates:
        is_valid = True
        for param, value in cand.items():
            if param in self.param_bounds:
                lo, hi = self.param_bounds[param]
                if not (lo <= value <= hi):
                    is_valid = False
                    logger.warning(f"候选点越界：{param}={value} ∉ [{lo}, {hi}]")
                    break
        if is_valid:
            valid.append(cand)

    return valid
```

---

## 6. 三个 Touchpoint 实现

### 6.1 Touchpoint 1a: 耦合矩阵

```python
# llm_interface.py:330-378
def generate_coupling_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成三个目标的耦合矩阵"""

    # Step 1: 渲染模板
    prompt = self.template_engine.render(
        "coupling_matrix",
        I1_RANGE=f"{self.param_bounds['I1'][0]} - {self.param_bounds['I1'][1]}",
        SOC1_RANGE=f"{self.param_bounds['SOC1'][0]} - {self.param_bounds['SOC1'][1]}",
        I2_RANGE=f"{self.param_bounds['I2'][0]} - {self.param_bounds['I2'][1]}",
        BATTERY_MODEL=self.battery_model,
        Q_NOM="5.0",
        SOC0="0.035",
        SOC_END="0.80",
    )

    # Step 2: 调用 LLM
    try:
        response = self.caller.call_with_retry(prompt)
        data = self.parser.extract_json(response)

        # Step 3: 解析矩阵
        W_time = np.array(data["W_time"])
        W_temp = np.array(data["W_temp"])
        W_aging = np.array(data["W_aging"])

        return W_time, W_temp, W_aging

    except Exception as e:
        logger.warning(f"LLM 耦合矩阵生成失败：{e}，使用回退策略")
        return self.fallback.generate_coupling_matrices()
```

### 6.2 Touchpoint 1b: Warmstart 候选

```python
# llm_interface.py:380-420
def generate_warmstart_candidates(self, n: int) -> np.ndarray:
    """生成 warmstart 候选点"""

    prompt = self.template_engine.render(
        "warmstart_candidates",
        I1_RANGE=f"{self.param_bounds['I1'][0]} - {self.param_bounds['I1'][1]}",
        SOC1_RANGE=f"{self.param_bounds['SOC1'][0]} - {self.param_bounds['SOC1'][1]}",
        I2_RANGE=f"{self.param_bounds['I2'][0]} - {self.param_bounds['I2'][1]}",
        BATTERY_MODEL=self.battery_model,
        Q_NOM="5.0",
        SOC0="0.035",
        SOC_END="0.80",
        NUM_CANDIDATES=str(n),
    )

    try:
        response = self.caller.call_with_retry(prompt)
        candidates = self.parser.extract_json(response)
        candidates = self.parser.validate_candidates(candidates)

        # 转换为数组
        X = np.array([[c["I1"], c["SOC1"], c["I2"]] for c in candidates])
        return X

    except Exception as e:
        logger.warning(f"LLM warmstart 生成失败：{e}，使用回退策略")
        return self.fallback.generate_warmstart_candidates(n)
```

### 6.3 Touchpoint 2: 迭代候选

```python
# llm_interface.py:422-500
def generate_iteration_candidates(
    self,
    n: int,
    state_dict: Dict[str, Any],
) -> np.ndarray:
    """基于当前优化状态生成候选点"""

    # 从 state_dict 提取信息
    iteration = state_dict["iteration"]
    theta_best = state_dict["theta_best"]
    f_min = state_dict["f_min"]
    mu = state_dict["mu"]
    sigma = state_dict["sigma"]
    stagnation_count = state_dict["stagnation_count"]
    w_vec = state_dict["w_vec"]
    sensitivity_info = state_dict.get("sensitivity_info", "")

    prompt = self.template_engine.render(
        "iterative_candidates",
        I1_RANGE=f"{self.param_bounds['I1'][0]} - {self.param_bounds['I1'][1]}",
        SOC1_RANGE=f"{self.param_bounds['SOC1'][0]} - {self.param_bounds['SOC1'][1]}",
        I2_RANGE=f"{self.param_bounds['I2'][0]} - {self.param_bounds['I2'][1]}",
        BATTERY_MODEL=self.battery_model,
        Q_NOM="5.0",
        SOC0="0.035",
        SOC_END="0.80",
        ITERATION=str(iteration),
        MAX_ITERATIONS=str(state_dict["max_iterations"]),
        BEST_I1=f"{theta_best[0]:.3f}",
        BEST_SOC1=f"{theta_best[1]:.3f}",
        BEST_I2=f"{theta_best[2]:.3f}",
        BEST_FTCH=f"{f_min:.6f}",
        MU_VALUES=f"[{mu[0]:.3f}, {mu[1]:.3f}, {mu[2]:.3f}]",
        SIGMA_VALUES=f"[{sigma[0]:.3f}, {sigma[1]:.3f}, {sigma[2]:.3f}]",
        STAGNATION_COUNT=str(stagnation_count),
        W_TIME=f"{w_vec[0]:.3f}",
        W_TEMP=f"{w_vec[1]:.3f}",
        W_AGING=f"{w_vec[2]:.3f}",
        NUM_CANDIDATES=str(n),
        SENSITIVITY_INFO=sensitivity_info,
    )

    try:
        response = self.caller.call_with_retry(prompt)
        candidates = self.parser.extract_json(response)
        candidates = self.parser.validate_candidates(candidates)

        X = np.array([[c["I1"], c["SOC1"], c["I2"]] for c in candidates])
        return X

    except Exception as e:
        logger.warning(f"LLM 迭代候选生成失败：{e}，使用回退策略")
        return self.fallback.generate_iteration_candidates(n, state_dict)
```

---

## 7. 物理启发式回退

### 7.1 回退策略的必要性

**LLM 可能失败的原因**：
- API 不可用（网络/配额）
- 响应格式错误
- 超时
- 生成不合理内容

### 7.2 回退实现

```python
# llm_interface.py:503-570
class PhysicsHeuristicFallback:
    """基于物理启发式的回退策略"""

    def __init__(self, param_bounds: Dict[str, Tuple[float, float]]):
        self.param_bounds = param_bounds
        self.rng = np.random.default_rng(42)

    def generate_coupling_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """回退：使用对角矩阵（无耦合假设）"""
        logger.info("使用回退策略：对角耦合矩阵")
        return np.eye(3), np.eye(3), np.eye(3)

    def generate_warmstart_candidates(self, n: int) -> np.ndarray:
        """回退：拉丁超立方采样 + 物理启发式中心点"""
        logger.info("使用回退策略：拉丁超立方采样")

        # 生成边界数组
        lo = np.array([self.param_bounds["I1"][0],
                       self.param_bounds["SOC1"][0],
                       self.param_bounds["I2"][0]])
        hi = np.array([self.param_bounds["I1"][1],
                       self.param_bounds["SOC1"][1],
                       self.param_bounds["I2"][1]])

        # 拉丁超立方采样
        X = self._latin_hypercube(n, 3)
        X = lo + X * (hi - lo)

        # 注入几个物理启发式点
       启发式点 = [
            [4.0, 0.35, 2.0],  # 保守
            [5.5, 0.40, 2.5],  # 平衡
            [6.5, 0.25, 3.5],  # 激进
        ]
        for i, pt in enumerate(启发式点[:min(3, n)]):
            X[i] = pt

        return X

    def generate_iteration_candidates(
        self,
        n: int,
        state_dict: Dict[str, Any],
    ) -> np.ndarray:
        """回退：μ±σ高斯采样"""
        logger.info("使用回退策略：高斯采样")

        mu = state_dict["mu"]
        sigma = state_dict["sigma"]

        # 截断高斯采样
        X = self.rng.normal(loc=mu, scale=sigma, size=(n, 3))

        # 裁剪到边界
        lo = np.array([self.param_bounds["I1"][0],
                       self.param_bounds["SOC1"][0],
                       self.param_bounds["I2"][0]])
        hi = np.array([self.param_bounds["I1"][1],
                       self.param_bounds["SOC1"][1],
                       self.param_bounds["I2"][1]])
        X = np.clip(X, lo, hi)

        return X

    def _latin_hypercube(self, n: int, d: int) -> np.ndarray:
        """拉丁超立方采样实现"""
        # ... 实现省略
        pass
```

---

## 8. 理解检查

完成本讲后，你应该能够：

- [ ] 描述三个 Touchpoint 的调用时机
- [ ] 解释模板引擎的工作原理
- [ ] 说明多后端 LLM 调用的实现方式
- [ ] 描述 JSON 提取的三种策略
- [ ] 解释回退策略的必要性

---

## 9. 课后练习

### 练习 4-1: 模板渲染

```python
from llm_interface import TemplateEngine

engine = TemplateEngine(template_dir="New_LLMBO/templates")

prompt = engine.render(
    "warmstart_candidates",
    I1_RANGE="3.0 - 7.0",
    SOC1_RANGE="0.10 - 0.70",
    I2_RANGE="1.0 - 5.0",
    BATTERY_MODEL="LG M50",
    Q_NOM="5.0",
    SOC0="0.035",
    SOC_END="0.80",
    NUM_CANDIDATES="5",
)

print(prompt)
```

观察输出的 prompt，找出所有被替换的占位符。

### 练习 4-2: Mock 测试

```python
from llm_interface import build_llm_interface

BOUNDS = {"I1": (3.0, 7.0), "SOC1": (0.1, 0.7), "I2": (1.0, 5.0)}

# 使用 Mock 后端
llm = build_llm_interface(
    param_bounds=BOUNDS,
    backend="mock",  # Mock 模式
)

# 测试 warmstart 生成
candidates = llm.generate_warmstart_candidates(n=5)
print(f"Warmstart 候选点:\n{candidates}")

# 测试耦合矩阵
W_time, W_temp, W_aging = llm.generate_coupling_matrices()
print(f"W_time:\n{W_time}")
```

### 练习 4-3: 真实 API 测试（可选）

如果有 OpenAI API key：

```python
llm = build_llm_interface(
    param_bounds=BOUNDS,
    backend="openai",
    model="gpt-3.5-turbo",
    api_key="sk-...",
)

candidates = llm.generate_warmstart_candidates(n=5)
print(candidates)
```

---

## 10. 预习下一讲

下一讲我们将学习 **采集函数系统**，了解：
- EI 期望改进的计算
- μ动态漂移追踪
- σ敏感度引导机制
- 停滞检测与扩张

**预习任务**：
1. 阅读 `acquisition.py` 的 `EICalculator` 类
2. 查看 `SearchMuTracker` 和 `SearchSigmaTracker`
3. 思考：为什么要用 EI×W 而不是 EI+W？

---

*第 04 讲 完*
