class ParseModelMessages:
    def __init__(self,response):
        self.response=response

    def get_messages(self):
        return self.response['messages'][0].content
    def tokens_num(self,):
        print("\n📊 Token 使用统计：")
        print(f"  总 tokens: {cb.total_tokens}")
        print(f"  输入 tokens: {cb.prompt_tokens}")
        print(f"  输出 tokens: {cb.completion_tokens}")