import random
from datasets import load_dataset
 
from arabic_sentiment.preprocessing import ArabicPreprocessor
from arabic_sentiment.language_model import NgramLanguageModel
from arabic_sentiment.naive_bayes import NaiveBayesClassifier
from arabic_sentiment.evaluation import accuracy, precision_recall_f1, confusion_matrix_str
def load_data():
    """Load the dataset and return train/test splits."""
    ds = load_dataset("arbml/Arabic_Sentiment_Twitter_Corpus")
    return ds["train"], ds["test"]

def run_preprocessing_demo(preprocessor: ArabicPreprocessor, raw_tweets: list) -> None:
    """Print a before/after table for 5 sample tweets."""
    print("\n" + "=" * 60)
    print("PART 1 — Preprocessing ")
    print("=" * 60)
    print(f"{'Before':<45} {'After'}")
    print("-" * 80)
    for tweet in raw_tweets[:5]:
        cleaned = preprocessor.preprocess(tweet, tokenize=False)
        print(f"{tweet[:45]:<45} {cleaned}")


def run_language_model(train_tokens_raw: list, train_tokens_clean: list, test_tokens_raw: list, test_tokens_clean: list) -> None:
    print("\n" + "=" * 60)
    print("PART 2 — Language Model Perplexity")
    print("=" * 60)

    models = {}
    print(f"\n{'Model':<10} | {'Raw PPL':>10} | {'Preprocessed':>16}")
    print("-" * 44)

    for n, name in [(2, "Bigram"), (3, "Trigram")]:
        raw_model = NgramLanguageModel(n=n)
        raw_model.train(train_tokens_raw)
        raw_ppl = raw_model.perplexity(test_tokens_raw)

        clean_model = NgramLanguageModel(n=n)
        clean_model.train(train_tokens_clean)
        clean_ppl = clean_model.perplexity(test_tokens_clean)

        print(f"{name:<10} | {raw_ppl:>10.2f} | {clean_ppl:>16.2f}")
        models[name] = clean_model

    print("\n" + "-" * 44)
    print("Generated Samples (preprocessed models):")
    print("-" * 44)
    for name, model in models.items():
        print(f"\n{name}:")
        for _ in range(3):
            print(" -", model.generate())

def run_naive_bayes(
    train_docs: list,
    train_labels: list,
    test_sample_docs: list,
    test_sample_labels: list,
    raw_test_tweets: list
) -> None:
    """Train NB classifier and print evaluation results."""
    print("\n" + "=" * 60)
    print("PART 3 — Naive Bayes Classifier")
    print("=" * 60)

    classifier = NaiveBayesClassifier()
    classifier.train(train_docs, train_labels)
    predictions = classifier.predict(test_sample_docs)

    acc = accuracy(predictions, test_sample_labels)
    prec, rec, f1 = precision_recall_f1(predictions, test_sample_labels)

    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1:        {f1:.3f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix_str(predictions, test_sample_labels, ["positive", "negative"]))

    print("\n5 Correct Predictions:")
    count = 0
    for tweet, pred, true in zip(raw_test_tweets, predictions, test_sample_labels):
        if pred == true and count < 5:
            print(f"  [{true}] {tweet[:60]}")
            count += 1

    print("\n5 Incorrect Predictions:")
    count = 0
    for tweet, pred, true in zip(raw_test_tweets, predictions, test_sample_labels):
        if pred != true and count < 5:
            print(f"  [true: {true} | pred: {pred}] {tweet[:60]}")
            count += 1

    print("\nTop features per class:")
    top = classifier.top_features(n=10)
    for label, words in top.items():
        print(f"\n  {label}:")
        for word, score in words:
            print(f"    {word}: {score:.3f}")


def run_library_baseline(
    train_docs: list,
    train_labels: list,
    test_sample_docs: list,
    test_sample_labels: list
) -> None:
    """Train sklearn pipeline and compare to scratch implementation."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report

    print("\n" + "=" * 60)
    print("PART 4 — Library Baseline (sklearn)")
    print("=" * 60)

    # sklearn needs strings not lists
    train_strings = [" ".join(doc) for doc in train_docs]
    test_strings  = [" ".join(doc) for doc in test_sample_docs]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("nb",    MultinomialNB())
    ])
    pipeline.fit(train_strings, train_labels)
    predictions = pipeline.predict(test_strings)

    print(classification_report(test_sample_labels, predictions))


def main() -> None:
    random.seed(42)

    print("=" * 60)
    print("Arabic Sentiment Analysis — Full Pipeline")
    print("=" * 60)

    # Step 1: Load data
    train_data, test_data = load_data()
    preprocessor = ArabicPreprocessor()

    raw_train_tweets = [item["tweet"] for item in train_data]
    raw_test_tweets  = [item["tweet"] for item in test_data]
    
    train_labels     = ["positive" if item["label"] == 1 else "negative" for item in train_data]
    test_labels      = ["positive" if item["label"] == 1 else "negative" for item in test_data]

    #  2: Preprocessing 
    run_preprocessing_demo(preprocessor, raw_train_tweets)

    #  3: Preprocess all data
    train_docs = [preprocessor.preprocess(t) for t in raw_train_tweets]
    test_docs  = [preprocessor.preprocess(t) for t in raw_test_tweets]

    #  4: Language model
    train_docs_raw = [doc.split() for doc in raw_train_tweets]
    test_docs_raw  = [doc.split() for doc in raw_test_tweets]
    run_language_model(train_docs_raw, train_docs, test_docs_raw[:500], test_docs[:500])

    #  5: Sample 100 for evaluation
    indices = random.sample(range(len(test_docs)), 100)
    test_sample_docs   = [test_docs[i]        for i in indices]
    test_sample_labels = [test_labels[i]      for i in indices]
    test_sample_tweets = [raw_test_tweets[i]  for i in indices]

    #  6: Naive Bayes
    run_naive_bayes(train_docs, train_labels,
                    test_sample_docs, test_sample_labels,
                    test_sample_tweets)

    #  7: Library baseline
    run_library_baseline(train_docs, train_labels,
                         test_sample_docs, test_sample_labels)


if __name__ == "__main__":
    main()