from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import math

Label = str  # positive or negative
class NaiveBayesClassifier:
    """
    Multinomial Naïve Bayes classifier for text.
    Supports Laplace (add-k) smoothing. Operates on pre-tokenized input.

    Attributes:
        k:                    Smoothing parameter (default 1.0).
        class_log_priors:     log P(c) for each class.
        word_log_likelihoods: log P(w | c) for each class and word.
        vocab:                All words seen during training.
    """

    def __init__(self, k: float = 1.0):
        self.k = k
        self.class_log_priors = {}
        self.word_log_likelihoods = defaultdict(dict)
        self.vocab = set()

    def train(
        self,
        documents: List[List[str]],
        labels: List[Label]
    ) -> None:
        """
        Estimate log priors and log likelihoods from training data.

        Steps:
            1. Count documents per class -> compute log priors.
            2. Concatenate all tokens per class -> build per-class word counts.
            3. Apply Laplace smoothing -> compute log likelihoods.
        """
        # 1: count docs per class
        total_docs = len(documents)
        class_doc_counts = Counter(labels)

        # log prior = log(docs in class / total docs)
        for label, count in class_doc_counts.items():
            self.class_log_priors[label] = math.log(count / total_docs)

        # 2: collect all words per class
        class_word_counts = defaultdict(Counter)
        for doc, label in zip(documents, labels):
            for word in doc:
                self.vocab.add(word)
                class_word_counts[label][word] += 1

        # 3: Laplace smoothing -> log likelihoods
        vocab_size = len(self.vocab)
        for label in class_doc_counts:
            total_words = sum(class_word_counts[label].values())
            for word in self.vocab:
                count = class_word_counts[label][word]
                self.word_log_likelihoods[label][word] = math.log(
                    (count + self.k) / (total_words + self.k * vocab_size)
                )

    def predict_one(self, tokens: List[str]) -> Label:
        """
        Predict the class of a single tokenized document.
        Skips unknown words silently.
        """
        scores = {}
        for label, log_prior in self.class_log_priors.items():
            score = log_prior
            for word in tokens:
                if word in self.vocab:
                    score += self.word_log_likelihoods[label][word]
            scores[label] = score
        return max(scores, key=scores.get)

    def predict(self, documents: List[List[str]]) -> List[Label]:
        """Predict classes for a list of tokenized documents."""
        return [self.predict_one(doc) for doc in documents]

    def top_features(self, n: int = 20) -> Dict[Label, List[Tuple[str, float]]]:
        """
        Top-n most discriminative words per class.
        Score = log P(w | class) - log P(w | other class)
        """
        
        labels = list(self.class_log_priors.keys())
        result = {}
        for label in labels:
            other = [l for l in labels if l != label][0]
            scores = []
            for word in self.vocab:
                score = (self.word_log_likelihoods[label][word]
                         - self.word_log_likelihoods[other][word])
                scores.append((word, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            result[label] = scores[:n]
        return result