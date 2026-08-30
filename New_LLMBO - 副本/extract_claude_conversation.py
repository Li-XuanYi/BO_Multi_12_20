"""
Claude 网页对话提取脚本
从 Claude.ai 网页对话中提取对话内容并保存为 JSON 格式

使用方法:
    python extract_claude_conversation.py <conversation_url>

示例:
    python extract_claude_conversation.py https://claude.ai/chats/xxxxx
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path


def extract_conversation_from_clipboard():
    """
    从剪贴板提取对话内容
    适用于用户手动复制整个对话页面的情况
    """
    import subprocess

    try:
        # Windows
        result = subprocess.run(['clip', '-o'], capture_output=True, text=True)
        return result.stdout
    except Exception:
        try:
            # macOS
            result = subprocess.run(['pbpaste'], capture_output=True, text=True)
            return result.stdout
        except Exception:
            # Linux
            result = subprocess.run(['xclip', '-selection', 'clipboard', '-o'],
                                   capture_output=True, text=True)
            return result.stdout


def extract_from_html_content(content):
    """
    从网页 HTML 或文本内容中提取对话

    支持两种模式：
    1. HTML 内容 - 尝试解析 HTML 标签
    2. 纯文本 - 按用户/Claude 标识符分割
    """
    import re
    from html.parser import HTMLParser

    conversations = []

    # 方法 1: HTML 模式 - 尝试从 HTML 结构中提取
    html_patterns = [
        (r'<div[^>]*class="[^"]*user[^"]*"[^>]*>(.*?)</div>', 'user'),
        (r'<div[^>]*class="[^"]*assistant[^"]*"[^>]*>(.*?)</div>', 'assistant'),
        (r'<div[^>]*class="[^"]*claude[^"]*"[^>]*>(.*?)</div>', 'assistant'),
        (r'<div[^>]*data-role="user"[^>]*>(.*?)</div>', 'user'),
        (r'<div[^>]*data-role="assistant"[^>]*>(.*?)</div>', 'assistant'),
    ]

    for pattern, role in html_patterns:
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        for match in matches:
            text = re.sub(r'<[^>]+>', '', match)
            text = text.strip()
            if text and text not in [c['content'] for c in conversations]:
                conversations.append({
                    'role': role,
                    'content': text,
                    'timestamp': datetime.now().isoformat()
                })

    # 方法 2: 纯文本模式 - 如果没有找到 HTML 结构，尝试文本模式
    if not conversations:
        # 按 "User:" 和 "Claude:" 或 "Assistant:" 分割
        text_patterns = [
            (r'(?:^|\n)\s*User:\s*(.+?)(?=(?:\n\s*(?:Claude|Assistant|User):)|$)', 'user'),
            (r'(?:^|\n)\s*(?:Claude|Assistant):\s*(.+?)(?=(?:\n\s*(?:User|Claude|Assistant):)|$)', 'assistant'),
        ]

        for pattern, role in text_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                text = match.strip()
                if text and text not in [c['content'] for c in conversations]:
                    conversations.append({
                        'role': role,
                        'content': text,
                        'timestamp': datetime.now().isoformat()
                    })

    # 按在原文中的位置排序（简单启发式）
    # 如果消息是乱序的，可以尝试根据内容重新排序
    return conversations


def generate_bookmarklet():
    """
    生成一个书签小程序，用户可以在浏览器中点击提取对话
    """
    bookmarklet = '''
javascript:(function(){
    var messages = [];
    var elements = document.querySelectorAll('[data-message]');

    elements.forEach(function(el) {
        var role = el.getAttribute('data-role') || 'unknown';
        var content = el.innerText.trim();
        var time = el.getAttribute('data-time') || new Date().toISOString();

        if(content) {
            messages.push({
                role: role,
                content: content,
                timestamp: time
            });
        }
    });

    if(messages.length === 0) {
        // 尝试其他选择器
        var userMsgs = document.querySelectorAll('.user-message, [class*="user"]');
        var assistantMsgs = document.querySelectorAll('.assistant-message, [class*="assistant"], [class*="claude"]');

        userMsgs.forEach(function(el) {
            var content = el.innerText.trim();
            if(content) {
                messages.push({role: 'user', content: content});
            }
        });

        assistantMsgs.forEach(function(el) {
            var content = el.innerText.trim();
            if(content) {
                messages.push({role: 'assistant', content: content});
            }
        });
    }

    var output = JSON.stringify({
        source: 'claude_web',
        url: window.location.href,
        extracted_at: new Date().toISOString(),
        message_count: messages.length,
        messages: messages
    }, null, 2);

    console.log(output);

    // 创建下载链接
    var blob = new Blob([output], {type: 'application/json'});
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'claude_conversation_' + Date.now() + '.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
})();
    '''.strip()

    return bookmarklet


def create_selenium_extractor(url, api_key=None):
    """
    使用 Selenium 自动提取对话内容

    需要安装: pip install selenium webdriver-manager
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
    except ImportError:
        print("需要安装 selenium: pip install selenium webdriver-manager")
        return None

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(url)

        # 等待页面加载
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # 滚动到页面底部以加载所有内容
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        # 获取页面内容
        page_source = driver.page_source

        return page_source

    finally:
        driver.quit()


def main():
    print("=" * 60)
    print("Claude 网页对话提取工具")
    print("=" * 60)

    if len(sys.argv) > 1:
        url = sys.argv[1]
        print(f"\n目标 URL: {url}")
    else:
        url = input("请输入 Claude 对话 URL: ").strip()

    # 创建输出目录
    output_dir = Path("extracted_conversations")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"claude_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    print("\n" + "=" * 60)
    print("提取方法选择:")
    print("1. 使用书签小程序 (推荐 - 最可靠)")
    print("2. 从剪贴板粘贴内容")
    print("3. 使用 Selenium 自动提取 (需要登录)")
    print("=" * 60)

    method = input("\n请选择提取方法 (1/2/3): ").strip()

    result = {
        "source": "claude_web_extractor",
        "url": url,
        "extracted_at": datetime.now().isoformat(),
        "messages": []
    }

    if method == "1":
        # 书签小程序方法
        bookmarklet = generate_bookmarklet()
        print("\n" + "=" * 60)
        print("书签小程序生成成功!")
        print("=" * 60)
        print("\n使用说明:")
        print("1. 在浏览器中打开书签管理器")
        print("2. 创建新书签，名称任意 (如 'Extract Claude Conversation')")
        print("3. 将以下代码粘贴到 URL/地址栏字段:")
        print("-" * 60)
        print(bookmarklet)
        print("-" * 60)
        print("\n4. 打开 Claude 对话页面")
        print("5. 点击刚创建的书签")
        print("6. JSON 文件会自动下载")
        print("\n或者，你也可以手动复制页面内容，然后选择选项 2")

    elif method == "2":
        # 剪贴板方法
        print("\n" + "=" * 60)
        print("方法 2: 从剪贴板或手动粘贴提取")
        print("=" * 60)
        print("\n【步骤 1】在浏览器中:")
        print("   1. 打开 Claude 对话页面 (https://claude.ai)")
        print("   2. 按 Ctrl+A 全选 (或 Cmd+A)")
        print("   3. 按 Ctrl+C 复制 (或 Cmd+C)")
        print("\n【步骤 2】按回车键从剪贴板读取，或直接粘贴内容:")

        input("按回车键尝试从剪贴板读取...")

        content = extract_conversation_from_clipboard()

        if not content or len(content) < 50:
            print("\n剪贴板内容太少或无法读取，请手动粘贴对话内容:")
            print("(粘贴后按 Ctrl+D 或输入 'EOF' 结束)\n")
            lines = []
            try:
                while True:
                    line = input()
                    if line.strip() == 'EOF':
                        break
                    lines.append(line)
                content = '\n'.join(lines)
            except EOFError:
                content = '\n'.join(lines)

        messages = extract_from_html_content(content)
        result["messages"] = messages
        result["message_count"] = len(messages)

    elif method == "3":
        # Selenium 方法
        print("\n注意：此方法需要您已登录 Claude 账户")
        print("正在启动浏览器...")

        page_source = create_selenium_extractor(url)
        if page_source:
            messages = extract_from_html_content(page_source)
            result["messages"] = messages
            result["message_count"] = len(messages)
        else:
            print("Selenium 提取失败")
            return
    else:
        print("无效的选择")
        return

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n提取完成!")
    print(f"共提取 {result.get('message_count', 0)} 条消息")
    print(f"结果已保存到：{output_file}")

    # 显示预览
    if result.get("messages"):
        print("\n消息预览:")
        print("-" * 40)
        for i, msg in enumerate(result["messages"][:5]):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:100]
            print(f"[{i+1}] {role}: {content}...")
        if len(result["messages"]) > 5:
            print(f"... 还有 {len(result['messages']) - 5} 条消息")


if __name__ == "__main__":
    main()
