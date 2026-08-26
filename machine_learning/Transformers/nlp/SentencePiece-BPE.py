import re
from collections import defaultdict



class SimpleBPETokenizer:
    """
    @ClassName:
    @Description:
    @Author:
    @Date:
    """
    def __init__(self, vocab_size, unk_token="[UNK]", pad_token="[PAD]"):

        """

        :param vocab_size:
        :param unk_token: 训练时没见过,推理时出现
        :param pad_token: 补齐长度
        """

        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.merges = {}  # 记录合并规则，按学习顺序排列


    def _get_stats(self, corpus):
        """统计所有相邻符号对的频率
        :param corpus:
        :return:
        """

        pairs = defaultdict(int)
        for tokens in corpus:
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs


    def _apply_merge(self, corpus, pair, new_token):
        """在整个语料库中应用一次合并
        :param corpus:
        :param pair:
        :param new_token:
        :return:
        """
        new_corpus = []
        for tokens in corpus:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_corpus.append(new_tokens)
        return new_corpus


    def train(self, texts):
        """训练 BPE
        :param texts:
        """
        # 1. 初始化：每个句子变成一个字符列表
        #    "我爱吃苹果" -> ['我', '爱', '吃', '苹', '果']
        corpus = [list(text) for text in texts]
        # 2. 获取基础词汇表（所有单字）
        base_vocab = set()
        for tokens in corpus:
            base_vocab.update(tokens)
        base_vocab = sorted(list(base_vocab))
        print(f"基础词汇数: {len(base_vocab)}")

        # 3. 开始合并循环
        self.merges = {}  # 重置
        current_vocab_size = len(base_vocab)

        while current_vocab_size < self.vocab_size:
            # 统计当前频率
            pairs = self._get_stats(corpus)
            if not pairs:
                break

            # 找到频率最高的 pair
            best_pair = max(pairs, key=pairs.get)

            """
            当频率相同时，会随机选择一个进行合并，但是
            当频率相同时，不按频率选择，而是按字符的 Unicode 码点顺序来选择。
            这样就能保证，每次训练相同的数据，都会得到完全一样的词表，而且是确定性的、可复现的。
            语料库太小，更长语义的词汇没有被合并，
            """

            # max_freq = max(pairs.values())
            #
            # # 2. 找出所有达到这个最高频率的 pair
            # candidates = [pair for pair, freq in pairs.items() if freq == max_freq]
            #
            # # 3. 对候选者排序：按第一个字排序，再按第二个字排序，确保确定性
            # #    这模拟了按Unicode码点排序，保证每次结果一致
            # best_pair = sorted(candidates, key=lambda x: (x[0], x[1]))[0]

            # 如果最高频只有1次，再合并也没意义了
            if pairs[best_pair] == 1:
                break

            # 创建新 token 并记录合并规则
            new_token = "".join(best_pair)
            self.merges[best_pair] = new_token
            print(f"合并 {best_pair} -> {new_token}，频率: {pairs[best_pair]}")

            # 在语料中应用这次合并
            corpus = self._apply_merge(corpus, best_pair, new_token)
            current_vocab_size += 1

        # 4. 构建最终词表
        self.vocab = [self.pad_token, self.unk_token] + base_vocab.copy()
        for merged_token in self.merges.values():
            if merged_token not in self.vocab:
                self.vocab.append(merged_token)

        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}
        print(f"训练完毕，最终词表大小: {len(self.vocab)}")
        print(f"合并规则: {self.merges}")


    def tokenize(self, text):
        """对单句进行分词"""
        # 1. 拆成单字
        tokens = list(text)

        # 2. 严格按训练时学到的顺序，逐条应用所有合并规则
        for pair, merged in self.merges.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens


    def encode(self, text):
        tokens = self.tokenize(text)
        unk_id = self.word2idx[self.unk_token]
        return [self.word2idx.get(t, unk_id) for t in tokens]


    def decode(self, ids):
        tokens = [self.idx2word.get(i, self.unk_token) for i in ids if i != self.word2idx[self.pad_token]]
        return "".join(tokens)


# --- 测试 ---
corpus = [
    "我每天都要吃苹果",
    "苹果很好吃",
    "我喜欢吃红苹果",
    "香蕉也是水果",
    "苹果和香蕉我都爱吃",
    "吃香蕉",
    "苹果汁很好喝",
]

tokenizer = SimpleBPETokenizer(vocab_size=30)  # 设大一点，看完整合并过程
tokenizer.train(corpus)

print("\n--- 最终词表 ---")
print(tokenizer.vocab)

print("\n--- 测试分词 ---")
test = "我今天吃了苹果和香蕉"
print(f"原文: {test}")
print(f"分词: {tokenizer.tokenize(test)}")
print(f"ID: {tokenizer.encode(test)}")
