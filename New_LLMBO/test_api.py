"""
API 测试脚本 - 验证新API配置是否正常工作
"""
import os
from openai import OpenAI

# 新API配置
API_KEY = "sk-7MaTMMMYCQtdisiY69eeoJF6oadNCJiF6JZz9bDif5Jacxc6"
BASE_URL = "https://api.chat.csu.edu.cn/v1"
MODEL = "deepseek-v3-thinking"

def test_api():
    """测试API连接"""
    print("=" * 60)
    print("API 连接测试")
    print("=" * 60)
    print(f"模型: {MODEL}")
    print(f"API地址: {BASE_URL}")
    print(f"API密钥: {API_KEY[:20]}...")
    print()

    try:
        client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
        )

        print("发送测试请求...")
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'API test successful' and nothing else."}
            ],
            temperature=0.0,
            max_tokens=50,
        )

        print("\n" + "=" * 60)
        print("测试成功!")
        print("=" * 60)
        print(f"响应: {response.choices[0].message.content}")
        print(f"模型: {response.model}")
        print(f"Token使用: {response.usage.total_tokens}")
        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print("测试失败!")
        print("=" * 60)
        print(f"错误: {e}")
        return False

if __name__ == "__main__":
    success = test_api()
    exit(0 if success else 1)
