from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

def evaluate_model_performance(true_labels, predictions):
    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    print("\nClassification Report:\n", classification_report(true_labels, predictions))
    print("\nConfusion Matrix:\n", confusion_matrix(true_labels, predictions))
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
