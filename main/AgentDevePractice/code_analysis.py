import os
import asyncio
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from model_factory.model_factory import ModelFactory

# ============================================
# 配置：确保已启动 start-coder
# ============================================
LLM =ModelFactory().create_model(local_model_type="coder")

# ============================================
# 核心函数
# ============================================
def read_code_file(file_path: str) -> str:
    """读取代码文件"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"错误: 文件 '{file_path}' 不存在"
    except Exception as e:
        return f"读取文件失败: {e}"

def analyze_code(code: str, language: str = "Python") -> dict:
    """让模型分析代码，返回结构化结果"""
    sys_prompt = """你是代码审查专家。请分析代码并以 JSON 格式返回：
{
  "summary": "代码功能概述",
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"],
  "priority": "高/中/低",
  "complexity": "简单/中等/复杂"
}
只返回 JSON，不要有其他内容。"""

    user_prompt = f"语言: {language}\n代码:\n{code}"

    response = LLM.invoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_prompt)
    ])

    # 提取 JSON
    content = response.content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return json.loads(content)

def generate_fixed_code(code: str, analysis: dict, language: str = "Python") -> str:
    """根据分析结果生成改进后的代码"""
    sys_prompt = """你是代码优化专家。根据分析建议生成改进后的完整代码。
要求：
1. 保持原有功能不变
2. 应用所有合理建议
3. 添加注释说明改动
4. 只返回代码，不要解释"""

    user_prompt = f"""语言: {language}
原始代码:
{code}

分析结果:
{json.dumps(analysis, ensure_ascii=False, indent=2)}

请生成改进后的完整代码。"""

    response = LLM.invoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_prompt)
    ])

    content = response.content.strip()
    if "```" in content:
        parts = content.split("```")
        for i in range(1, len(parts), 2):
            if parts[i].strip().startswith("python") or parts[i].strip().startswith("py"):
                parts[i] = parts[i].split("\n", 1)[1] if "\n" in parts[i] else parts[i]
            return parts[i]
    return content

def generate_comparison(original: str, fixed: str) -> str:
    """生成对比报告"""
    sys_prompt = "你是技术文档专家。对比原始代码和改进代码，生成简要对比报告。"
    user_prompt = f"""原始代码:
{original}

改进后代码:
{fixed}

请输出:
1. 主要改进点 (3-5条)
2. 性能/可读性提升
3. 最终评价"""

    response = LLM.invoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_prompt)
    ])
    return response.content

# ============================================
# 主流程
# ============================================
async def main():
    # 测试代码
    test_code = """def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left, middle, right = [], [], []
    for x in arr:
        if x < pivot: left.append(x)
        elif x == pivot: middle.append(x)
        else: right.append(x)
    return quick_sort(left) + middle + quick_sort(right)

data = [3, 6, 8, 10, 1, 2, 1]
print(quick_sort(data))"""

    print("="*60)
    print("🚀 代码分析与自动修复 Agent")
    print("="*60)

    print("\n📖 原始代码:")
    print("-"*40)
    print(test_code)
    print("-"*40)

    print("\n🔍 正在分析代码...")
    analysis = analyze_code(test_code)

    print("\n📊 分析结果:")
    print(f"  功能: {analysis.get('summary', 'N/A')}")
    print(f"  复杂度: {analysis.get('complexity', 'N/A')}")
    print(f"  优先级: {analysis.get('priority', 'N/A')}")
    print(f"  问题: {', '.join(analysis.get('issues', ['无']))}")
    print(f"  建议: {', '.join(analysis.get('suggestions', ['无']))}")

    print("\n🛠️ 正在生成改进代码...")
    fixed_code = generate_fixed_code(test_code, analysis)

    print("\n✅ 改进后代码:")
    print("-"*40)
    print(fixed_code)
    print("-"*40)

    print("\n📋 对比报告:")
    print("-"*40)
    comparison = generate_comparison(test_code, fixed_code)
    print(comparison)
    print("-"*40)

    print("\n✨ 完成！")

if __name__ == "__main__":
    asyncio.run(main())