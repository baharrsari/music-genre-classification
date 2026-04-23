import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = "viridis"


def plot_class_distribution(df, label_col="genre"):
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = df[label_col].value_counts().sort_index()
    colors = sns.color_palette(PALETTE, len(counts))
    counts.plot(kind="bar", ax=ax, color=colors)
    ax.set_title("Müzik Türü Dağılımı (GTZAN)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Tür")
    ax.set_ylabel("Dosya Sayısı")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_single_model_metrics(metrics, model_name, dataset_name="Test"):
    classification_metrics = {
        "Accuracy": metrics.get("accuracy"),
        "Prec (macro)": metrics.get("precision_macro"),
        "Prec (weighted)": metrics.get("precision_weighted"),
        "Recall (macro)": metrics.get("recall_macro"),
        "Recall (weighted)": metrics.get("recall_weighted"),
        "F1 (macro)": metrics.get("f1_macro"),
        "F1 (weighted)": metrics.get("f1_weighted"),
        "Specificity": metrics.get("specificity_macro"),
    }

    reliability_metrics = {
        "ROC-AUC (OvR)": metrics.get("roc_auc_ovr"),
        "Cohen's Kappa": metrics.get("cohen_kappa"),
        "MCC": metrics.get("mcc"),
    }

    extra_metrics = {}
    if metrics.get("log_loss") is not None:
        extra_metrics["Log Loss"] = metrics["log_loss"]
    if metrics.get("train_time") is not None:
        extra_metrics["Train Time (s)"] = metrics["train_time"]

    n_panels = 3 if extra_metrics else 2

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    axes = list(axes)

    fig.suptitle(
        f"{model_name} — {dataset_name} Set Metrikleri",
        fontsize=15, fontweight="bold", y=1.02
    )

    # Panel 1: Sınıflandırma
    ax1 = axes[0]
    names1 = list(classification_metrics.keys())
    vals1 = [v if v is not None else 0 for v in classification_metrics.values()]
    colors1 = sns.color_palette("Blues_d", len(names1))
    bars1 = ax1.barh(names1, vals1, color=colors1)
    ax1.set_xlim(0, 1.05)
    ax1.set_title("Sınıflandırma Metrikleri", fontsize=12, fontweight="bold")
    ax1.invert_yaxis()
    for bar, val in zip(bars1, vals1):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=9)

    # Panel 2: Güvenilirlik
    ax2 = axes[1]
    names2 = list(reliability_metrics.keys())
    vals2 = [v if v is not None else 0 for v in reliability_metrics.values()]
    colors2 = sns.color_palette("Greens_d", len(names2))
    bars2 = ax2.barh(names2, vals2, color=colors2)
    ax2.set_xlim(0, 1.05)
    ax2.set_title("Güvenilirlik Metrikleri", fontsize=12, fontweight="bold")
    ax2.invert_yaxis()
    for bar, val in zip(bars2, vals2):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=9)

    # Panel 3: Log Loss & Süre
    if n_panels == 3:
        ax3 = axes[2]
        names3 = list(extra_metrics.keys())
        vals3 = list(extra_metrics.values())
        colors3 = sns.color_palette("Oranges_d", len(names3))
        bars3 = ax3.barh(names3, vals3, color=colors3)
        ax3.set_title("Log Loss & Süre", fontsize=12, fontweight="bold")
        ax3.invert_yaxis()
        for bar, val in zip(bars3, vals3):
            ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names, model_name="Model", dataset_name="Test"):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_title(f"{model_name} — {dataset_name} Confusion Matrix",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.show()


def plot_model_comparison(all_results, metric_keys=None, title=None):
    if metric_keys is None:
        metric_keys = [
            "accuracy", "precision_macro", "recall_macro",
            "f1_macro", "cohen_kappa", "mcc"
        ]
    model_names = list(all_results.keys())
    data = []
    for model_name in model_names:
        for mk in metric_keys:
            val = all_results[model_name].get(mk, 0)
            if val is None:
                val = 0
            data.append({"Model": model_name, "Metric": mk, "Value": val})
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(max(12, len(model_names) * 2.5), 7))
    sns.barplot(data=df, x="Model", y="Value", hue="Metric", ax=ax, palette="husl")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Skor", fontsize=12)
    ax.set_title(title or "Model Performans Karşılaştırması", fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    plt.tight_layout()
    plt.show()


def plot_model_comparison_heatmap(all_results, metric_keys=None, title=None):
    if metric_keys is None:
        metric_keys = [
            "accuracy", "precision_macro", "precision_weighted",
            "recall_macro", "recall_weighted", "f1_macro", "f1_weighted",
            "specificity_macro", "roc_auc_ovr", "log_loss",
            "cohen_kappa", "mcc", "train_time"
        ]
    data = {}
    for model_name, metrics in all_results.items():
        col = {}
        for mk in metric_keys:
            val = metrics.get(mk)
            col[mk] = val if val is not None else np.nan
        data[model_name] = col
    df = pd.DataFrame(data).T
    fig, ax = plt.subplots(figsize=(max(14, len(metric_keys) * 1.2), max(4, len(data) * 1.2)))
    sns.heatmap(df.astype(float), annot=True, fmt=".4f", cmap="YlGnBu",
                ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title(title or "Model Karşılaştırma Heatmap'i", fontsize=14, fontweight="bold")
    ax.set_ylabel("Model")
    ax.set_xlabel("Metrik")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_training_time_comparison(all_results, title=None):
    names, times = [], []
    for model_name, metrics in all_results.items():
        t = metrics.get("train_time")
        if t is not None:
            names.append(model_name)
            times.append(t)
    if not names:
        print("Hiçbir modelde train_time bilgisi yok.")
        return
    sorted_pairs = sorted(zip(times, names))
    times_sorted = [p[0] for p in sorted_pairs]
    names_sorted = [p[1] for p in sorted_pairs]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("rocket", len(names))
    bars = ax.barh(names_sorted, times_sorted, color=colors)
    ax.set_xlabel("Eğitim Süresi (saniye)", fontsize=12)
    ax.set_title(title or "Model Eğitim Süreleri", fontsize=14, fontweight="bold")
    for bar, t in zip(bars, times_sorted):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{t:.2f}s", va="center", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_log_loss_comparison(all_results, title=None):
    names, losses = [], []
    for model_name, metrics in all_results.items():
        ll = metrics.get("log_loss")
        if ll is not None:
            names.append(model_name)
            losses.append(ll)
    if not names:
        print("Hiçbir modelde log_loss bilgisi yok.")
        return
    sorted_pairs = sorted(zip(losses, names))
    losses_sorted = [p[0] for p in sorted_pairs]
    names_sorted = [p[1] for p in sorted_pairs]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("flare", len(names))
    bars = ax.barh(names_sorted, losses_sorted, color=colors)
    ax.set_xlabel("Log Loss (düşük = daha iyi)", fontsize=12)
    ax.set_title(title or "Model Log Loss Karşılaştırması", fontsize=14, fontweight="bold")
    for bar, ll in zip(bars, losses_sorted):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{ll:.4f}", va="center", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_all_confusion_matrices(all_results, class_names, dataset_name="Test"):
    n = len(all_results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()
    for idx, (name, metrics) in enumerate(all_results.items()):
        cm = metrics["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                    xticklabels=class_names, yticklabels=class_names)
        axes[idx].set_title(name, fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle(f"Confusion Matrix Karşılaştırması ({dataset_name})",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_roc_curves(models_dict, X, y_true, label_encoder):
    n_classes = len(label_encoder.classes_)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("husl", len(models_dict))
    for (name, model), color in zip(models_dict.items(), colors):
        if not hasattr(model, "predict_proba"):
            print(f"  {name}: predict_proba desteklenmiyor, atlanıyor.")
            continue
        y_proba = model.predict_proba(X)
        fpr_list, tpr_list, auc_list = [], [], []
        for i in range(n_classes):
            fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            auc_i = auc(fpr_i, tpr_i)
            fpr_list.append(fpr_i)
            tpr_list.append(tpr_i)
            auc_list.append(auc_i)
        all_fpr = np.unique(np.concatenate(fpr_list))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr_list[i], tpr_list[i])
        mean_tpr /= n_classes
        macro_auc = np.mean(auc_list)
        ax.plot(all_fpr, mean_tpr, color=color, linewidth=2,
                label=f"{name} (AUC = {macro_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Rastgele (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Eğrileri — Macro Average", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_baseline_vs_gridsearch(models, baseline, gridsearch, title="Baseline vs GridSearch — Test Accuracy"):
    """
    Modellerin baseline ve gridsearch doğruluklarını karşılaştırır.
    """
    x = range(len(models))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - 0.2 for i in x], baseline, 0.4, label="Baseline", color="steelblue")
    ax.bar([i + 0.2 for i in x], gridsearch, 0.4, label="GridSearch", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0.4, 0.85)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.show()
