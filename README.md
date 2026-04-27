# Arabic Sentiment Analysis project 1 - AraNLP Bootcamp 

## How to run
pip install datasets scikit-learn
python main.py

## Results

### Perplexity
| Model   | Raw PPL   | Preprocessed |
|---------|-----------|------------------|
| Bigram  | 24651.40  | 24234.07         |
| Trigram | 38618.46  | 32317.27         |

### Classification Metrics (100 samples)

| Metric    | Naive Bayes (scratch) | sklearn baseline |
|-----------|-----------------------|------------------|
| Accuracy  | 0.780                 | 0.790            |
| Precision | 0.814                 | 0.830            |
| Recall    | 0.714                 | 0.710            |
| F1        | 0.761                 | 0.770            |

## Reflections

### - Does preprocessing help? For which metric and why? (Why does it lower perplexity?)
Yes, Preprocessing lowered perplexity for both models
(Bigram: 24651 → 24234, Trigram: 38618 → 32317).

Preprocessing lowers perplexity because it reduces vocabulary size 
words like "أحمد" and "احمد" become the same token after normalization,
so the model sees more examples per word and makes better predictions

A lower perplexity means the model is less confused when it sees new
text it can predict the next word more confidently. In our case,
the preprocessed model is slightly more confident than the raw model.

### - Which n-gram order worked better and why?
Bigram performed better with lower perplexity (24234 vs 32317)
Trigram model does not find enough context to learn from because tweets are short this lead resulting higher perplexity

### - How does your scratch implementation compare to sklearn's?
Results are very close , scratch reached 78% accuracy while sklearn reached 79%
The small gap is because TF-IDF down-weights common words automatically, while our scratch model treats all words equally

### - What surprised you about Arabic Twitter text?
The extra use of character elongation " Tatweel" like -> جميييييل , also the emojis. All this show for us how important preprocessing steps are for informal
Arabic text before building any model
