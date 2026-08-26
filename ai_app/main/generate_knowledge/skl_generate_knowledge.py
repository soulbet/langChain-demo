import os
import re
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tqdm import tqdm

from ai_app.main.generate_knowledge.knowledge_map import UNSUPERVISED_TOPICS

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
# 监督学习知识点列表（根据你的目录生成）
# ============================================

topic=UNSUPERVISED_TOPICS


# ============================================
# 核心函数
# ============================================
def load_agent_prompt(agent_name: str = "engineering-ai-engineer") -> str:
    """从 agency-agents 加载专家代理定义"""
    current_dir = Path(r"../../agency-agents").resolve()
    agent_file = Path(f"{current_dir}/engineering/{agent_name}.md")
    if agent_file.exists():
        return agent_file.read_text(encoding="utf-8")
    else:
        # 回退到默认提示词
        return "你是一位资深机器学习专家，擅长用清晰、结构化的方式解释复杂概念。"


def generate_topic_note(topic_name: str, category: str) -> str:
    """调用模型生成知识点的 Markdown 笔记"""

    # 1. 加载专家代理
    agent_prompt = load_agent_prompt("ai-engineer")  # 或者 "data-scientist"

    # 2. 构建完整提示词
    structure_requirements = r"""
请严格按照以下结构输出 Markdown 格式的笔记：
1. **核心定义**：用 1-2 句话给出定义。
2. **数学原理/公式**：使用标准的 LaTeX 语法描述关键公式。
   - 行间公式用 $$ ... $$ 包裹。
   - 行内公式用 $ ... $ 包裹。
3. **关键要点**：列出 3-5 个核心要点（用 `-` 列表）。
4. **代码示例**：给出一个简洁的 sklearn 代码示例。
5. **面试常见问题**：列出 2-3 个可能的面试题。
6. **优缺点**：简要总结优缺点。
7. **常用使用场景**：列举 2-3 个典型应用场景。
"""

    full_system_prompt = agent_prompt + "\n\n" + structure_requirements

    user_prompt = f"""知识点：{topic_name}
分类：{category}

请生成该知识点的面试复习笔记。"""

    response = LLM.invoke([
        SystemMessage(content=full_system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return response.content


def save_markdown(content: str, topic_name: str, output_dir: str = "./supervised_learning_notes"):
    """保存 Markdown 到文件"""
    os.makedirs(output_dir, exist_ok=True)
    # 生成安全的文件名
    safe_name = re.sub(r'[^\w\-\.]', '_', topic_name)
    filepath = os.path.join(output_dir, f"{safe_name}.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ 已保存: {filepath}")


# ============================================
# 主流程
# ============================================
def main():
    output_dir = "unsupervised_learning_notes"
    print("=" * 60)
    print("📚 监督学习知识点总结生成器")
    print("=" * 60)
    print(f"📁 输出目录: {output_dir}")
    print(f"📄 共 {len(topic)} 个知识点")

    for topic_name, category in tqdm(topic, desc="生成知识点总结"):
        try:
            note_content = generate_topic_note(topic_name, category)
            save_markdown(note_content, category, output_dir)
        except Exception as e:
            print(f"❌ 生成失败 {topic_name}: {e}")

    print("\n✨ 全部完成！")


if __name__ == "__main__":
    main()