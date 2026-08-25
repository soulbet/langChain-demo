# 加载训练集
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files=r"D:\work_space\model\datasets\industry_instruction_semantic_cluster_dedup_医疗_train.jsonl"
)
data={'instruction': '', 'id': 'industry-instruction-edu', 'conversations': [{'from': 'human', 'value': '问：右侧桥小区囊性占位，囊肿可能大。偏肢体发育不良，巨脑回；右侧桥小脑区囊性占位，囊肿可能大.曾经治疗情况和效果：左手左脚没有右面灵活，爱往左歪脖，曾经感冒发烧39度抽过3、4次，抽时症状：身体僵硬呕吐。不知道是和囊肿有关还是和发育不良有关'}, {'from': 'gpt', 'value': '答：根据你描述的症状和病史，右侧桥小区囊性占位可能是导致偏肢体发育不良和巨脑回的原因之一。囊肿的大小可能也会对症状产生影响。\n\n至于感冒发烧抽搐的症状，这可能与囊肿有关，也可能与发育不良有关。囊肿对中枢神经系统的正常功能可能会有影响，导致肌肉僵硬和呕吐等症状。\n\n然而，请注意，我只是凭借你提供的信息做出初步推断，而不能提供确切的诊断。为了明确病因和确定最佳治疗方案，我建议你尽快咨询专业的医生进行详细的检查和进一步的评估。他们可以根据你的具体情况给出准确的诊断和建议。'}, {'from': 'human', 'value': '问：现在这个囊肿就已经压迫脑干了，如果以后再长大把脑干都压上了以后手术是不是就很难很危险了'}, {'from': 'gpt', 'value': '答：如果囊肿已经压迫了脑干，那么情况确实比较复杂和危险。手术可能会变得更具风险，因为脑干是一个重要的神经结构，手术过程中需要小心以避免额外的损伤。\n\n不过，你的具体情况需要医生进行详细的评估和决策。专业医生将会考虑多个因素，包括囊肿的位置、大小、对周围结构的影响以及你的整体健康状况等等。他们会权衡手术的利弊，并与你详细讨论手术风险和可能的好处。\n\n最重要的是，请务必在专业医生的指导下进行进一步的治疗和管理。他们能够为你提供最佳的个性化建议，并指导你选择最适合你情况的治疗方案。'}], 'lang': 'zh'}

def extract_first_turn(data):
    conversations=data['conversations']
    result=[]
    if len(conversations) >=2:
        for index in range(0,len(conversations), 2):
            question=conversations[index]["value"].replace('问：','').strip()
            answer=conversations[index+1]["value"].replace('问：','').strip()
            result.append({"instruction": question, "output": answer})
    return result

def format_instruction(data):
    """将数据格式化为 Qwen 的聊天模板"""
    messages = [
        {"role": "user", "content": data["instruction"]},
        {"role": "assistant", "content": data["output"]}
    ]
    tokenizer.apply_chat
extract_dataset=extract_first_turn(dataset)


