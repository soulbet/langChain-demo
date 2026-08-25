class SimpleChineseTokenizer:
    def __init__(self):
        # 1. 定义特殊token，[PAD]用来补齐句子长度，[UNK]代表不认识的字
        self.special_tokens = ["[PAD]", "[UNK]"]
        # 2. 用一个字典来存储“字到编号”的映射
        self.char2idx = {}
        # 3. 用一个列表来存储“编号到字”的映射
        self.idx2char = []

    def fit(self, texts):
        """从语料中学习词表，texts是句子列表"""
        # 收集所有出现过的字
        chars = set()
        for text in texts:
            for char in text:
                chars.add(char)

        # 按固定顺序排好：先放特殊token，再放收集到的字
        vocab = self.special_tokens + sorted(list(chars))

        # 建立映射关系
        self.char2idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx2char = vocab
        print(f"词表大小: {len(vocab)}")  # 打印词表大小

    def encode(self, text):
        """把句子变成编号序列"""
        ids = []
        for char in text:
            # 如果字在词表里，就取它的编号；否则用[UNK]的编号(这里是1)
            ids.append(self.char2idx.get(char, self.char2idx["[UNK]"]))
        return ids

    def decode(self, ids):
        """把编号序列变回句子"""
        chars = []
        for idx in ids:
            # 忽略填充符[PAD]
            if idx == self.char2idx["[PAD]"]:
                continue
            chars.append(self.idx2char[idx])
        return "".join(chars)


# --- 测试一下 ---
corpus = [
    "我特别爱看月亮",
    "我也爱吃西瓜",
    "月亮和西瓜我都爱"
]

tokenizer = SimpleChineseTokenizer()
tokenizer.fit(corpus)  # 学习词表

print(tokenizer.char2idx)
# 输出类似：{'[PAD]':0, '[UNK]':1, '。':2, '和':3, '我':4, '月':5, '瓜':6, '特':7, '爱':8, '看':9, '西':10, '别':11, '亮':12, '吃':13, '也':14}

test_sentence = "我爱看月亮"
print(tokenizer.encode(test_sentence))
# 输出：[4, 8, 9, 5, 12]

print(tokenizer.decode([4, 8, 9, 5, 12]))
# 输出：我爱看月亮