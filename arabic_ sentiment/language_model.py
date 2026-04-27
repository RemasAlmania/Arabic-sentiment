from collections import defaultdict
from typing import List, Tuple, Literal
import math
import random

NGramOrder = Literal[2, 3]

class NgramLanguageModel:
    """
    A bigram or trigram language model with Laplace (add-1) smoothing.

    Attributes:
        n:              The order of the model (2 for bigram, 3 for trigram).
        vocab:          The set of all known tokens.
        counts:         Raw n-gram counts.
        context_counts: Counts of (n-1)-gram contexts.
    """

    def __init__(self, n: NGramOrder = 2):
        self.n = n
        self.vocab = set()
        self.counts = defaultdict(int)
        self.context_counts = defaultdict(int)

    def _extract_ngrams(self, tokens: List[str]) -> List[Tuple[str, ...]]:
        """
        Extract all n-grams from a token list.
        Adds <s> start and </s> end tokens.
        Bigram:  ['a','b'] -> [('<s>','a'), ('a','b'), ('b','</s>')]
        Trigram: ['a','b'] -> [('<s>','<s>','a'), ('<s>','a','b'), ('a','b','</s>')]
        """
        padding = ['<s>'] * (self.n - 1)
        padded = padding + tokens + ['</s>']
        ngrams = []
        for i in range(len(padded) - self.n + 1):
            ngrams.append(tuple(padded[i:i + self.n]))
        return ngrams

    def train(self, corpus: List[List[str]]) -> None:
        """
        Train on a list of tokenized sentences.
        
        Args:
            corpus: A list of token lists (one per tweet/sentence).
        """
        for tokens in corpus:
            for token in tokens:
                self.vocab.add(token)
            for ngram in self._extract_ngrams(tokens):
                self.counts[ngram] += 1
                self.context_counts[ngram[:-1]] += 1

    def log_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Log probability of an n-gram using Laplace smoothing.
        P(w | context) = (count(context, w) + 1) / (count(context) + |V|)
        Returns log base 2 of the probability.
        """
        context = ngram[:-1]
        numerator = self.counts[ngram] + 1
        denominator = self.context_counts[context] + len(self.vocab)
        return math.log2(numerator / denominator)

    def sentence_log_probability(self, tokens: List[str]) -> float:
        """       
        Return the total log probability of a tokenized sentence.
        
        This is the sum of log probabilities of each n-gram in the sentence.
        """
        total = 0
        for ngram in self._extract_ngrams(tokens):
            total += self.log_probability(ngram)
        return total

    def perplexity(self, corpus: List[List[str]]) -> float:
        """
        Perplexity = 2^(-average log probability per token)
        Lower perplexity = better model.
        """
        total_log_prob = 0
        total_tokens = 0
        for tokens in corpus:
            total_log_prob += self.sentence_log_probability(tokens)
            total_tokens += len(tokens)
        return 2 ** (-total_log_prob / total_tokens)

    def generate(self, seed: List[str] = None, max_tokens: int = 20) -> str:
        """
        Generate tokens using the model until </s> or max_tokens.
        Uses random.choices weighted by probability.
        
        """
        context = ['<s>'] * (self.n - 1)
        if seed:
            context = seed[-(self.n - 1):]
        result = []
        for _ in range(max_tokens):
            candidates = list(self.vocab) + ['</s>']
            weights = []
            for word in candidates:
                ngram = tuple(context) + (word,)
                weights.append(2 ** self.log_probability(ngram))
            next_word = random.choices(candidates, weights=weights)[0]
            if next_word == '</s>':
                break
            result.append(next_word)
            context = (context + [next_word])[-(self.n - 1):]
        return ' '.join(result)