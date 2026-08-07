import asyncio

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from model_factory.model_factory import ModelFactory

# ============================================
# 配置：确保已启动 start-coder
# ============================================
LLM = ChatOpenAI(
    base_url="http://localhost:55001/v1",
    api_key="sk-dummy",
    model="qwen2.5-coder-7b",
    temperature=0.2
)
llm = ModelFactory().create_model(local_model_type="agent")
async def test_minimal():
    print("=" * 60)
    print("🧪 最小化测试：模型完全自由生成")
    print("=" * 60)

    # 最简提示词，没有任何格式约束

    print("\n📤 发送请求...")
    latex_response = LLM.invoke("请用 LaTeX 写出 Yeo-Johnson 变换的公式。")
    # 第二步：要求模型将 LaTeX 转译为纯文本
    plain_text_response = llm.invoke(f"请将以下 LaTeX 公式用纯文本转述：{latex_response}")
    print(plain_text_response)

    print("\n📥 模型回复:")
    print("-" * 40)
    print(plain_text_response.content)
    print("-" * 40)

if __name__ == "__main__":
    asyncio.run(test_minimal())