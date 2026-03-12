"""
llmbo/components/llm_client.py
================================
统一 LLM 客户端与速率限制器

设计目标:
1. 统一接口 - 将所有 LLM 访问集中到一个客户端
2. 速率限制 - 实现双阶段限制（预估 token + 请求后校正）
3. 多后端支持 - OpenAI / Anthropic / Ollama / Mock
4. 自动重试 - 指数退避重试机制
5. 并发控制 - 限制并发请求数

借鉴 LLAMBO 的 RateLimiter 思路:
- 第一阶段：预估 token 数量，判断是否超过剩余配额
- 第二阶段：实际请求后，根据实际使用 token 数校正配额

用法示例:
    from llmbo.components.llm_client import LLMClient, RateLimiter, LLMConfig

    # 创建配置
    config = LLMConfig(
        api_key="sk-xxx",
        base_url="https://api.example.com/v1",
        model="gpt-4o",
        rate_limit_tokens=100000,
        rate_limit_requests=500,
    )

    # 创建客户端
    client = LLMClient(config)

    # 发送请求
    response = await client.chat(
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
    )

    # 带速率限制的批量请求
    responses = await client.batch_chat(
        messages_list=[...],
        max_concurrent=5,
    )
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# §A  配置数据类
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LLMConfig:
    """LLM 客户端配置"""

    # API 配置
    api_key: str = "sk-default"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"

    # 速率限制
    rate_limit_tokens: int = 100000  # 每分钟 token 数
    rate_limit_requests: int = 500   # 每分钟请求数

    # 重试配置
    max_retries: int = 3
    retry_base_delay: float = 1.0  # 指数退避基数（秒）
    retry_max_delay: float = 60.0  # 最大延迟（秒）

    # 超时配置
    request_timeout: float = 120.0  # 请求超时（秒）

    # 并发控制
    max_concurrent_requests: int = 10

    # 日志
    verbose: bool = False

    def __post_init__(self):
        if self.rate_limit_tokens <= 0:
            raise ValueError("rate_limit_tokens 必须 > 0")
        if self.rate_limit_requests <= 0:
            raise ValueError("rate_limit_requests 必须 > 0")


class LLMBackend(Enum):
    """LLM 后端类型"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    MOCK = "mock"


# ═══════════════════════════════════════════════════════════════════════════
# §B  速率限制器
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TokenBucket:
    """Token 桶实现"""

    capacity: int  # 桶容量
    tokens: float = field(default=0.0)  # 当前 token 数
    last_update: float = field(default_factory=time.time)  # 上次更新时间
    refill_rate: float = field(default=0.0)  # 补充速率 (tokens/秒)

    def __post_init__(self):
        if self.tokens == 0.0:
            self.tokens = float(self.capacity)
        if self.refill_rate == 0.0:
            # 默认每分钟补满
            self.refill_rate = self.capacity / 60.0

    def consume(self, tokens: int) -> bool:
        """
        尝试消费 token

        Args:
            tokens: 需要消费的 token 数

        Returns:
            True 如果成功消费，False 如果 token 不足
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self):
        """补充 token"""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )
        self.last_update = now

    def wait_time(self, tokens: int) -> float:
        """
        计算需要等待的时间才能消费指定数量的 token

        Args:
            tokens: 需要消费的 token 数

        Returns:
            等待时间（秒）
        """
        self._refill()

        if self.tokens >= tokens:
            return 0.0

        needed = tokens - self.tokens
        return needed / self.refill_rate


class RateLimiter:
    """
    双阶段速率限制器

    实现 LLAMBO 风格的限制策略:
    1. 第一阶段：预估 token + 检查配额
    2. 第二阶段：实际请求后校正配额
    """

    def __init__(
        self,
        token_limit: int,
        request_limit: int,
        window_seconds: int = 60,
    ):
        """
        初始化速率限制器

        Args:
            token_limit: 每分钟 token 限制
            request_limit: 每分钟请求数限制
            window_seconds: 时间窗口（秒）
        """
        self.token_bucket = TokenBucket(
            capacity=token_limit,
            refill_rate=token_limit / window_seconds,
        )
        self.request_bucket = TokenBucket(
            capacity=request_limit,
            refill_rate=request_limit / window_seconds,
        )
        self._window_seconds = window_seconds

        # 统计信息
        self._total_requests = 0
        self._total_tokens = 0
        self._rate_limited_count = 0

    def estimate_tokens(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
    ) -> int:
        """
        预估请求的 token 数量

        简单估算：每个字符约 0.25 个 token（英文）
        更精确的估算需要使用 tokenizer

        Args:
            messages: 消息列表
            max_tokens: 最大输出 token 数

        Returns:
            预估的总 token 数
        """
        # 输入 token 估算
        input_chars = sum(
            len(msg.get("content", ""))
            for msg in messages
        )
        estimated_input_tokens = int(input_chars * 0.25)

        # 总 token = 输入 + 输出
        return estimated_input_tokens + max_tokens

    async def acquire(
        self,
        estimated_tokens: int,
        timeout: float = 30.0,
    ) -> bool:
        """
        第一阶段：获取配额

        Args:
            estimated_tokens: 预估 token 数
            timeout: 超时时间（秒）

        Returns:
            True 如果成功获取配额
        """
        start_time = time.time()

        while True:
            # 检查请求配额
            if not self.request_bucket.consume(1):
                wait_time = self.request_bucket.wait_time(1)
                if time.time() - start_time + wait_time > timeout:
                    self._rate_limited_count += 1
                    return False
                await asyncio.sleep(min(wait_time, 1.0))
                continue

            # 检查 token 配额
            if not self.token_bucket.consume(estimated_tokens):
                wait_time = self.token_bucket.wait_time(estimated_tokens)
                if time.time() - start_time + wait_time > timeout:
                    self.request_bucket.tokens += 1  # 归还请求配额
                    self._rate_limited_count += 1
                    return False
                await asyncio.sleep(min(wait_time, 1.0))
                continue

            # 成功获取配额
            self._total_requests += 1
            return True

    def commit(
        self,
        actual_tokens: int,
        estimated_tokens: int,
    ):
        """
        第二阶段：校正配额

        Args:
            actual_tokens: 实际使用的 token 数
            estimated_tokens: 之前预估的 token 数
        """
        # 如果实际使用超过预估，需要补充差额
        if actual_tokens > estimated_tokens:
            diff = actual_tokens - estimated_tokens
            self.token_bucket.tokens += diff

        self._total_tokens += actual_tokens

    @property
    def stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "rate_limited_count": self._rate_limited_count,
            "token_bucket_available": int(self.token_bucket.tokens),
            "request_bucket_available": int(self.request_bucket.tokens),
        }


# ═══════════════════════════════════════════════════════════════════════════
# §C  LLM 响应数据类
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LLMResponse:
    """LLM 响应"""

    content: str
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    finish_reason: str = ""
    raw_response: Any = None

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", self.prompt_tokens + self.completion_tokens)


# ═══════════════════════════════════════════════════════════════════════════
# §D  LLM 客户端主类
# ═══════════════════════════════════════════════════════════════════════════

class LLMClient:
    """
    统一 LLM 客户端

    提供:
    - 多后端支持（OpenAI / Anthropic / Ollama / Mock）
    - 速率限制（双阶段）
    - 自动重试（指数退避）
    - 并发控制
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        backend: Optional[LLMBackend] = None,
    ):
        """
        初始化 LLM 客户端

        Args:
            config: 客户端配置
            backend: 后端类型（可选，自动检测）
        """
        self.config = config or LLMConfig()
        self.verbose = self.config.verbose

        # 自动检测后端
        if backend:
            self.backend = backend
        else:
            self.backend = self._detect_backend()

        # 速率限制器
        self.rate_limiter = RateLimiter(
            token_limit=self.config.rate_limit_tokens,
            request_limit=self.config.rate_limit_requests,
        )

        # 并发信号量
        self._semaphore = asyncio.Semaphore(
            self.config.max_concurrent_requests
        )

        # 客户端实例（懒加载）
        self._async_client: Any = None

        if self.verbose:
            print(f"[LLMClient] 初始化：backend={self.backend.value}, model={self.config.model}")

    def _detect_backend(self) -> LLMBackend:
        """根据配置自动检测后端"""
        url = self.config.base_url.lower()

        if "anthropic" in url:
            return LLMBackend.ANTHROPIC
        elif "ollama" in url or "localhost" in url or "127.0.0.1" in url:
            return LLMBackend.OLLAMA
        elif self.config.model.lower() == "mock":
            return LLMBackend.MOCK
        else:
            return LLMBackend.OPENAI

    def _get_client(self):
        """获取异步客户端（懒加载）"""
        if self._async_client is not None:
            return self._async_client

        if self.backend == LLMBackend.MOCK:
            self._async_client = None
        else:
            # OpenAI 兼容客户端（支持 OpenAI / Ollama）
            try:
                from openai import AsyncOpenAI
                self._async_client = AsyncOpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.request_timeout,
                )
            except ImportError:
                raise ImportError("需要安装 openai 包：pip install openai")

        return self._async_client

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> LLMResponse:
        """
        发送聊天请求

        Args:
            messages: 消息列表
            temperature: 采样温度
            max_tokens: 最大输出 token 数
            **kwargs: 其他参数

        Returns:
            LLM 响应
        """
        # 第一阶段：预估 token 并获取配额
        estimated_tokens = self.rate_limiter.estimate_tokens(messages, max_tokens)

        acquired = await self.rate_limiter.acquire(
            estimated_tokens,
            timeout=30.0,
        )

        if not acquired:
            raise RuntimeError("速率限制：无法获取配额")

        async with self._semaphore:
            try:
                response = await self._call_llm(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )

                # 第二阶段：校正配额
                self.rate_limiter.commit(
                    actual_tokens=response.total_tokens,
                    estimated_tokens=estimated_tokens,
                )

                return response

            except Exception as e:
                # 请求失败，归还配额
                self.rate_limiter.rate_limiter.token_bucket.tokens += estimated_tokens
                raise

    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> LLMResponse:
        """实际调用 LLM（带重试）"""
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                if self.backend == LLMBackend.MOCK:
                    return await self._mock_call(messages, max_tokens)
                elif self.backend in (LLMBackend.OPENAI, LLMBackend.OLLAMA):
                    return await self._openai_call(
                        messages, temperature, max_tokens, **kwargs
                    )
                elif self.backend == LLMBackend.ANTHROPIC:
                    return await self._anthropic_call(
                        messages, temperature, max_tokens, **kwargs
                    )
                else:
                    raise ValueError(f"未知后端：{self.backend}")

            except Exception as e:
                last_error = e

                if attempt < self.config.max_retries:
                    delay = min(
                        self.config.retry_base_delay * (2 ** attempt),
                        self.config.retry_max_delay,
                    )

                    if self.verbose:
                        print(
                            f"[LLMClient] 请求失败 (尝试 {attempt + 1}/"
                            f"{self.config.max_retries + 1}): {e}, "
                            f"等待 {delay:.1f}s"
                        )

                    await asyncio.sleep(delay)
                else:
                    break

        raise last_error

    async def _openai_call(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> LLMResponse:
        """OpenAI 兼容 API 调用"""
        client = self._get_client()

        response = await client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            model=response.model,
            finish_reason=response.choices[0].finish_reason,
            raw_response=response,
        )

    async def _anthropic_call(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> LLMResponse:
        """Anthropic API 调用"""
        try:
            import anthropic
        except ImportError:
            raise ImportError("需要安装 anthropic 包：pip install anthropic")

        client = anthropic.AsyncAnthropic(
            api_key=self.config.api_key,
        )

        # 转换消息格式
        system_message = ""
        chat_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_message = content
            else:
                chat_messages.append({
                    "role": "user" if role == "user" else "assistant",
                    "content": content,
                })

        response = await client.messages.create(
            model=self.config.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message,
            messages=chat_messages,
            **kwargs,
        )

        return LLMResponse(
            content=response.content[0].text,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            model=response.model,
            finish_reason=response.stop_reason,
            raw_response=response,
        )

    async def _mock_call(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
    ) -> LLMResponse:
        """Mock 调用（测试用）"""
        await asyncio.sleep(0.1)  # 模拟延迟

        return LLMResponse(
            content="[Mock Response]",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
            },
            model="mock",
            finish_reason="stop",
        )

    async def batch_chat(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ) -> List[LLMResponse]:
        """
        批量发送聊天请求（带并发控制）

        Args:
            messages_list: 消息列表的列表
            temperature: 采样温度
            max_tokens: 最大输出 token 数
            **kwargs: 其他参数

        Returns:
            响应列表
        """
        async def call_with_index(idx: int, messages: List[Dict[str, str]]):
            try:
                response = await self.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                return idx, response
            except Exception as e:
                if self.verbose:
                    print(f"[LLMClient] 批量请求 #{idx} 失败：{e}")
                return idx, None

        tasks = [
            call_with_index(idx, messages)
            for idx, messages in enumerate(messages_list)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 按原始顺序返回结果
        responses = [None] * len(messages_list)
        for result in results:
            if isinstance(result, tuple) and result[0] is not None:
                idx, response = result
                responses[idx] = response

        return responses

    @property
    def stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "backend": self.backend.value,
            "model": self.config.model,
            **self.rate_limiter.stats,
        }


# ═══════════════════════════════════════════════════════════════════════════
# §E  工厂函数
# ═══════════════════════════════════════════════════════════════════════════

def create_llm_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    rate_limit_tokens: int = 100000,
    rate_limit_requests: int = 500,
    verbose: bool = False,
) -> LLMClient:
    """
    快速创建 LLM 客户端

    Args:
        api_key: API 密钥
        base_url: API 基础 URL
        model: 模型名称
        rate_limit_tokens: token 速率限制
        rate_limit_requests: 请求速率限制
        verbose: 详细输出

    Returns:
        LLMClient 实例
    """
    config = LLMConfig(
        api_key=api_key or "sk-default",
        base_url=base_url or "https://api.openai.com/v1",
        model=model or "gpt-4o",
        rate_limit_tokens=rate_limit_tokens,
        rate_limit_requests=rate_limit_requests,
        verbose=verbose,
    )

    return LLMClient(config)


# ═══════════════════════════════════════════════════════════════════════════
# §F  CLI 测试
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import asyncio

    async def test():
        print("=" * 60)
        print("LLMClient 测试")
        print("=" * 60)

        # 测试 Mock 后端
        print("\n[测试 1] Mock 后端")
        client = create_llm_client(
            model="mock",
            rate_limit_tokens=1000,
            rate_limit_requests=100,
            verbose=True,
        )

        response = await client.chat(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
        )
        print(f"  响应：{response.content}")
        print(f"  Token 使用：{response.total_tokens}")

        # 测试批量请求
        print("\n[测试 2] 批量请求")
        messages_list = [
            [{"role": "user", "content": f"Message {i}"}]
            for i in range(5)
        ]

        responses = await client.batch_chat(messages_list)
        print(f"  成功响应：{sum(1 for r in responses if r is not None)}/{len(responses)}")

        # 显示统计
        print("\n[测试 3] 统计信息")
        stats = client.stats
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)

    asyncio.run(test())
