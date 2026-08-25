"""
最长匹配原则：这就是“正向最大匹配”的精髓。我们总是贪心地选择当前能匹配到的最长词语。比如词典里同时有“西瓜”和“西”，遇到“吃西瓜”时，系统会优先匹配“西瓜”，而不是“西”和“瓜”。

规则优先于统计：这里完全不涉及概率，全靠你提供的词典。分词的优劣完全取决于你的词典质量。

局限性：这种硬规则方法无法处理歧义。比如“研究生命的起源”，如果词典里有“研究生”、“生命”、“的”、“起源”，最大匹配就会错误地切成“研究生 / 命 / 的 / 起源”。

"""


class DictTokenizer:
    def __init__(self, word_dict):
        """
        word_dict: 自定义的词语列表，比如 ["深度学习", "学习", "深", "度", ...]
        """
        # 将词语列表转为集合，查询更快
        self.word_set = set(word_dict)
        # 添加特殊token
        self.special_tokens = ["[PAD]", "[UNK]"]
        # 找出词典中最长的词的长度，用于限制搜索窗口
        self.max_len = max(len(w) for w in self.word_set) if self.word_set else 0
        print(f"词典加载完毕，共 {len(self.word_set)} 个词，最长词长 {self.max_len}")

        # 构建词表映射
        vocab = self.special_tokens + sorted(list(self.word_set))
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}

    def tokenize(self, text):
        """核心：正向最大匹配分词"""
        tokens = []
        start = 0
        text_len = len(text)

        while start < text_len:
            matched_word = None
            # 从max_len开始，逐步缩短搜索窗口，寻找最长匹配词
            for window in range(self.max_len, 0, -1):
                end = start + window
                if end > text_len:
                    continue
                candidate = text[start:end]

                # 如果候选词在词典里，就是一个成功匹配
                if candidate in self.word_set:
                    matched_word = candidate
                    tokens.append(matched_word)
                    # 匹配成功后，窗口直接跳到词尾
                    start = end
                    break

            # 如果没有任何词语匹配，就把当前字当作[UNK]
            if matched_word is None:
                tokens.append(text[start])  # 按单字处理
                start += 1

        return tokens

    def encode(self, text):
        """分词后转ID"""
        tokens = self.tokenize(text)
        unk_id = self.word2idx["[UNK]"]
        return [self.word2idx.get(t, unk_id) for t in tokens]

    def decode(self, ids):
        """ID转回文本"""
        tokens = []
        for idx in ids:
            word = self.idx2word.get(idx, "[UNK]")
            if word == "[PAD]":
                continue
            tokens.append(word)
        # 直接拼接，简单处理
        return "".join(tokens)


# --- 测试 ---
# 自定义一个小词典，模拟词语概念
my_dict = ["深度", "学习", "深度强化学习", "强化", "很", "有趣", "有意思"]

tokenizer = DictTokenizer(my_dict)

text = "深度学习很有意思"
print(f"分词结果: {tokenizer.tokenize(text)}")
# 输出: 分词结果: ['深度', '学习', '很', '有意思']
# 注意：“有意思”比“有”长，所以优先匹配

text2 = "深度强化学习很有趣"
print(f"分词结果: {tokenizer.tokenize(text2)}")
# 输出: 分词结果: ['深度强化学习', '很', '有趣']
# 注意：最长的“深度强化学习”被成功匹配，而不是切成更短的词

print(f"ID序列: {tokenizer.encode(text)}")
# 输出类似: [4, 6, 7, 8]