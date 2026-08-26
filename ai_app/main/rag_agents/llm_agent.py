from langchain.agents import create_agent


class LlmAgent:
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt

    def create_rag_agent(self, llm, tools):
        """使用 LangChain v1.x 的 create_agent 创建 RAG Agent"""
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=self.system_prompt,
        )

        return agent