import os
import asyncio
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ============================================
# 配置：确保已启动 start-coder
# ============================================
LLM = ChatOpenAI(
    base_url="http://localhost:55001/v1",
    api_key="sk-dummy",
    model="qwen2.5-coder-7b",
    temperature=0.3
)

# ============================================
# 知识点映射 (文件名 → 知识点名称 & 分类)
# ============================================
TOPIC_MAP = {
    "归一化": {"name": "Min-Max 归一化", "category": "特征缩放"},
    "标准化": {"name": "Z-Score 标准化", "category": "特征缩放"},
    "缩放": {"name": "特征缩放", "category": "特征缩放"},
    "指定范围缩放": {"name": "指定范围缩放", "category": "特征缩放"},
    "稀疏数组缩放": {"name": "稀疏数组缩放", "category": "特征缩放"},
    "带离群值的数据缩放": {"name": "带离群值的数据缩放", "category": "特征缩放"},
    "核函数": {"name": "核函数", "category": "特征变换"},
    "特征二值化": {"name": "特征二值化", "category": "特征编码"},
    "特征编码": {"name": "特征编码", "category": "特征编码"},
    "离散化": {"name": "离散化", "category": "特征变换"},
    "非线性变换": {"name": "非线性变换", "category": "特征变换"},
}


def detect_topic(filename: str) -> tuple:
    """从文件名推断知识点名称和分类"""
    for key, value in TOPIC_MAP.items():
        if key in filename:
            return value["name"], value["category"]
    # 如果没有匹配，使用文件名本身
    name = Path(filename).stem.replace("_", " ").replace("-", " ")
    return name, "未分类"

def load_agent_prompt(agent_name: str = "data-scientist") -> str:
    agent_file = Path(f"./agency-agents/agents/{agent_name}.md")
    if agent_file.exists():
        return agent_file.read_text(encoding="utf-8")
    else:
        sys_prompt = r"""你是机器学习专家，擅长用简洁清晰的方式解释概念。
        请为给定的知识点生成一份结构化的纯文本总结，适合面试复习使用。

        **输出格式和内容要求（必须100%遵守）**：
        1. 用 LaTeX 写出符合 Markdown 标准的公式。要求用 $$ ... $$ 包裹行间公式，用 \dfrac 代替 \frac。

        输出结构如下
        ## 1. 核心定义
        用一两句话描述下概念定义


        ## 2. 数学原理/公式
        用文字描述公式
        **不要用 LaTeX 语法，直接用文字和符号表达。**

        ## 3. 关键要点
        列出 3-5 个核心要点（用 `-` 列表）。

        ## 4. 代码示例 (Python)
        给出一个简洁的 sklearn 或 numpy 示例。

        ## 5. 面试常见问题
        列出 2-3 个面试中可能问到的题目。

        ## 6. 优缺点
        - 优点: ...
        - 缺点: ...
        ## 7. 常用的使用场景
        """
        # 回退到默认提示词
        return sys_prompt

# ============================================
# 核心函数
# ============================================
def generate_knowledge_summary(topic_name: str, category: str, code_snippet: str = "") -> str:
    """调用模型生成知识点的 Markdown 总结"""

    # 如果有对应的代码文件，作为参考
    code_context = f"\n参考代码示例:\n```python\n{code_snippet}\n```" if code_snippet else ""
    sys_prompt = load_agent_prompt("data-scientist")


    user_prompt = f"""知识点: {topic_name}
分类: {category}
{code_context}

请生成该知识点的 Markdown 总结。"""

    response = LLM.invoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_prompt)
    ])
    return response.content


def scan_directory(directory: str) -> list:
    """扫描目录，返回 Python 文件列表及内容"""
    files_info = []
    for file in Path(directory).rglob("*.py"):
        # 跳过 __init__.py 等
        if file.name.startswith("_"):
            continue
        try:
            content = file.read_text(encoding="utf-8")
            files_info.append({
                "path": file,
                "name": file.name,
                "content": content
            })
        except Exception as e:
            print(f"⚠️ 无法读取 {file.name}: {e}")
    return files_info


def save_markdown(content: str, filename: str, output_dir: str = "./knowledge_summary"):
    """保存 Markdown 到文件"""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = filename.replace(".py", "").replace(" ", "_").replace("-", "_")
    filepath = os.path.join(output_dir, f"{safe_name}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ 已保存: {filepath}")


# ============================================
# 主流程
# ============================================
async def main():
    # 要扫描的目录 (请修改为你的实际路径)
    target_dir = r"/sklearn&matplot/sk-preprocessing_demo"  # 👈 改成你的目录

    print("=" * 60)
    print("📚 机器学习知识点总结生成器")
    print("=" * 60)
    print(f"📁 扫描目录: {target_dir}")

    files = scan_directory(target_dir)
    print(f"📄 找到 {len(files)} 个 Python 文件")

    if not files:
        print("❌ 未找到任何 Python 文件，请检查目录路径")
        return

    print("\n开始生成知识点总结...\n")

    for f in files:
        topic_name, category = detect_topic(f["name"])
        print(f"🔍 处理: {f['name']} → {topic_name} ({category})")

        try:
            summary = generate_knowledge_summary(topic_name, category, f["content"])
            save_markdown(summary, f["name"])
        except Exception as e:
            print(f"❌ 生成失败 {f['name']}: {e}")

    print("\n✨ 全部完成！")


if __name__ == "__main__":
    asyncio.run(main())