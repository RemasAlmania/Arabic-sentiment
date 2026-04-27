from typing import List, Tuple


def accuracy(predictions: List[str], gold: List[str]) -> float:
    """Compute accuracy as the fraction of correct predictions."""
    correct = 0
    for pred, true in zip(predictions, gold):
        if pred == true:
            correct += 1
    return correct / len(gold)


def precision_recall_f1(
    predictions: List[str],
    gold: List[str],
    positive_label: str = "positive"
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 for the positive class.

    Returns:
        (precision, recall, f1) as a tuple of floats.
    """
    
    tp = 0  # predicted positive actually positive
    fp = 0  # predicted positive actually negative
    fn = 0  # predicted negative actually positive

    for pred, true in zip(predictions, gold):
        if pred == positive_label and true == positive_label:
            tp += 1
        elif pred == positive_label and true != positive_label:
            fp += 1
        elif pred != positive_label and true == positive_label:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return precision, recall, f1


def confusion_matrix_str(
    predictions: List[str],
    gold: List[str],
    labels: List[str]
) -> str:
    """
    Return a pretty-printed confusion matrix string.

    Example:
              Pred Pos  Pred Neg
    True Pos     42        8
    True Neg      5       45
    """
 
    matrix = {}
    for true in labels:
        matrix[true] = {}
        for pred in labels:
            matrix[true][pred] = 0

    for pred, true in zip(predictions, gold):
        matrix[true][pred] += 1


    header = f"{'':12}" + "".join(f"Pred {l:8}" for l in labels)
    rows = ""
    for true in labels:
        row = f"True {true:8}"
        for pred in labels:
            row += f"{matrix[true][pred]:12}"
        rows += row + "\n"

    return header + "\n" + rows