# -*- coding: utf-8 -*-
"""
    @Project: langChain-demo
    @File   : SentencePiece-unigram.py
    @Author : zyf
    @Date   : 2026/8/26 22:08
    @Desc   : 
    """
import math
from collections import Counter


class UnigramTokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.vocab = {}  # 最终的词表: {token: log_prob}
        self.unk_token = "<unk>"
        self.max_word_len = 0

    def _get_all_substrings(self, word):
        """获取一个词的所有可能子串"""
        subs = set()
        n = len(word)
        for i in range(n):
            for j in range(i + 1, min(n + 1, i + self.max_word_len + 1)):
                subs.add(word[i:j])
        return subs

    def _initialize_vocab(self, corpus):
        """初始化候选词表：提取所有高频子串"""
        word_counts = Counter()
        for word in corpus.split():  # 假设语料已按空格分好，这里为了简化处理
            word_counts[word] += 1

        self.max_word_len = max(len(w) for w in word_counts.keys())

        # 收集所有可能的子串
        sub_counts = Counter()
        for word, count in word_counts.items():
            for sub in self._get_all_substrings(word):
                sub_counts[sub] += count

        # 初始词表：取出现次数大于 1 的子串，并加上单字符
        initial_vocab = [sub for sub, count in sub_counts.items() if count > 1]
        # 确保所有单字符都在词表中，避免无法切分
        for char in "".join(word_counts.keys()):
            if char not in initial_vocab:
                initial_vocab.append(char)

        return initial_vocab, word_counts

    def _viterbi_tokenize(self, word, vocab_probs):
        """维特比算法：寻找一个词的最优切分路径，最大化对数概率"""
        n = len(word)
        # dp[i] 表示切分到第 i 个字符时的最大对数概率
        dp = [-float('inf')] * (n + 1)
        # back[i] 记录达到 dp[i] 时的上一个切分点
        back = [None] * (n + 1)
        dp[0] = 0.0  # 空串的概率为 0 (log 1 = 0)

        for i in range(1, n + 1):
            for j in range(max(0, i - self.max_word_len), i):
                sub = word[j:i]
                prob = vocab_probs.get(sub, vocab_probs.get(self.unk_token))
                if dp[j] + prob > dp[i]:
                    dp[i] = dp[j] + prob
                    back[i] = j

        # 回溯找出最优路径
        tokens = []
        i = n
        while i > 0:
            j = back[i]
            tokens.append(word[j:i])
            i = j
        tokens.reverse()
        return tokens

    def _compute_word_prob(self, word, vocab_probs):
        """计算一个词在给定词表下的最大对数概率"""
        tokens = self._viterbi_tokenize(word, vocab_probs)
        return sum(vocab_probs.get(t, vocab_probs.get(self.unk_token)) for t in tokens)

    def train(self, corpus):
        print("开始训练 Unigram Tokenizer...")
        # 1. 初始化
        initial_vocab, word_counts = self._initialize_vocab(corpus)

        # 初始化概率：基于频次的简单归一化
        counts = Counter()
        for word, w_count in word_counts.items():
            for sub in self._get_all_substrings(word):
                if sub in initial_vocab:
                    counts[sub] += w_count

        total_counts = sum(counts.values())
        # 概率取对数，防止连乘下溢
        vocab_probs = {sub: math.log(c / total_counts) for sub, c in counts.items()}
        # 补上 unk 的概率
        vocab_probs[self.unk_token] = math.log(1e-10)

        # 2. 迭代剪枝
        while len(vocab_probs) > self.vocab_size:
            # --- E 步：计算语料似然度 ---
            # 计算当前词表下，整个语料的对数似然
            total_loss = 0
            word_losses = {}
            for word, count in word_counts.items():
                loss = self._compute_word_prob(word, vocab_probs)
                total_loss += loss * count
                word_losses[word] = loss

            # --- M 步：计算每个 token 的损失差异 ---
            # 如果删掉某个 token，整个语料的似然度会下降多少？
            token_importance = {}
            for token in list(vocab_probs.keys()):
                if token == self.unk_token:
                    continue

                # 假设去掉该 token，把它的概率设为极小
                temp_probs = vocab_probs.copy()
                temp_probs[token] = vocab_probs[self.unk_token]

                new_total_loss = 0
                for word, count in word_counts.items():
                    # 这里为了加速，只重算包含该 token 的词。简化版直接重算全部
                    new_loss = self._compute_word_prob(word, temp_probs)
                    new_total_loss += new_loss * count

                # 损失差异越小，说明这个 token 越没用（删了它对整体概率影响不大）
                token_importance[token] = total_loss - new_total_loss

            # 3. 剔除最不重要的 20% 的 token (保留单字符以防万一)
            sorted_tokens = sorted(token_importance.items(), key=lambda x: x[1])
            num_to_remove = max(1, int(0.2 * len(vocab_probs)))

            # 保护单字符不被删除
            removed = 0
            for token, loss in sorted_tokens:
                if removed >= num_to_remove:
                    break
                if len(token) > 1:  # 单字符不删
                    del vocab_probs[token]
                    removed += 1

            if removed == 0:  # 如果全剩单字符了，强行结束
                break

            # 4. 重新归一化概率
            total_prob = sum([math.exp(p) for t, p in vocab_probs.items() if t != self.unk_token])
            for t in vocab_probs:
                if t != self.unk_token:
                    vocab_probs[t] = math.log(math.exp(vocab_probs[t]) / total_prob)

        self.vocab = vocab_probs
        print(f"训练完成，最终词表大小: {len(self.vocab)}")

    def encode(self, text):
        """对输入文本进行编码"""
        tokens = []
        for word in text.split():
            word_tokens = self._viterbi_tokenize(word, self.vocab)
            tokens.extend(word_tokens)
        return tokens


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    # 模拟一段语料 (为了看出效果，重复一些词组)
    corpus = "小明 喜欢 打篮球 小明 喜欢 踢足球 小明 不喜欢 写作业 小明 喜欢 看电影"

    # 实例化并训练 (目标词表设为 15，以便看到合并效果)
    tokenizer = UnigramTokenizer(vocab_size=15)
    tokenizer.train(corpus)

    print("\n最终词表 (Token: Log Prob):")
    for t, p in sorted(tokenizer.vocab.items(), key=lambda x: x[1], reverse=True):
        print(f"'{t}': {p:.4f}")

    print("\n" + "=" * 50)
    print("分词测试")
    print("=" * 50)
    test_text = "小明 喜欢 玩游戏"
    tokens = tokenizer.encode(test_text)
    print(f"输入: {test_text}")
    print(f"分词结果: {tokens}")
