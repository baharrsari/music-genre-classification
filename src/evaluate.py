import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    log_loss,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import label_binarize


def compute_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(len(cm)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity)
    return float(np.mean(specificities))


def evaluate_model(model, X, y_true, label_encoder, train_time=None):
    n_classes = len(label_encoder.classes_)
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)
    else:
        y_proba = None

    metrics = {}

    metrics["accuracy"] = round(accuracy_score(y_true, y_pred), 4)

    metrics["precision_macro"] = round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4)
    metrics["precision_weighted"] = round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4)

    metrics["recall_macro"] = round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4)
    metrics["recall_weighted"] = round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4)

    metrics["f1_macro"] = round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4)
    metrics["f1_weighted"] = round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4)

    metrics["specificity_macro"] = round(compute_specificity(y_true, y_pred), 4)

    if y_proba is not None:
        try:
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            metrics["roc_auc_ovr"] = round(roc_auc_score(y_true_bin, y_proba, multi_class="ovr", average="macro"), 4)
        except ValueError:
            metrics["roc_auc_ovr"] = None
        try:
            metrics["log_loss"] = round(log_loss(y_true, y_proba), 4)
        except ValueError:
            metrics["log_loss"] = None
    else:
        metrics["roc_auc_ovr"] = None
        metrics["log_loss"] = None

    metrics["cohen_kappa"] = round(float(cohen_kappa_score(y_true, y_pred)), 4)
    metrics["mcc"] = round(float(matthews_corrcoef(y_true, y_pred)), 4)
    metrics["train_time"] = round(train_time, 4) if train_time is not None else None

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    metrics["classification_report"] = classification_report(
        y_true, y_pred, target_names=label_encoder.classes_, zero_division=0
    )

    return metrics


def collect_all_results(results_dict, dataset_name="Test"):
    exclude = {"confusion_matrix", "classification_report"}
    all_data = {}
    for model_name, metrics in results_dict.items():
        model_col = {}
        for key, value in metrics.items():
            if key not in exclude:
                model_col[key] = value
        all_data[model_name] = model_col
    df = pd.DataFrame(all_data)
    df.index.name = "Metric"
    return df
