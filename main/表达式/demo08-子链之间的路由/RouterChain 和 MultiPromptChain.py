"""
遗留方式

"""

from langchain.chains.router import LLMRouterChain, MultiPromptChain
from langchain.chains.router.llm_router import RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains import LLMChain, ConversationChain
from langchain.prompts import PromptTemplate

from main.model_factory import model_factory

model = model_factory().create_model()
# 定义多个目标链（物理、数学、英语）
physics_chain = LLMChain(llm=model, prompt=PromptTemplate.from_template("物理专家：{input}"))
math_chain = LLMChain(llm=model, prompt=PromptTemplate.from_template("数学专家：{input}"))
english_chain = LLMChain(llm=model, prompt=PromptTemplate.from_template("英语专家：{input}"))

# 构建目的地链字典
destinations = {"physics": physics_chain, "math": math_chain, "english": english_chain}

# 构建默认链
default_chain = ConversationChain(llm=model, output_key="text")

# 构建路由链
router_prompt = PromptTemplate(
    template=MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations="physics:物理类\nmath:数学类\nenglish:英语类"),
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm=model, prompt=router_prompt)

# 组合成多提示词链
multi_prompt_chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destinations,
    default_chain=default_chain,
    verbose=True
)

multi_prompt_chain.invoke("重力加速度是多少？")