# src/plot_finbert_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.paths import CSV_DIR, FIGURES_DIR

def main():
    history_path = CSV_DIR / "finbert_btc_direction_history.csv"
    preds_path   = CSV_DIR / "finbert_btc_direction_predictions.csv"
    report_path  = CSV_DIR / "finbert_btc_direction_classification_report.csv"

    history_df = pd.read_csv(history_path)
    preds_df   = pd.read_csv(preds_path)
    report_df  = pd.read_csv(report_path, index_col=0)

    print("History:")
    print(history_df)
    print("\nPredições (head):")
    print(preds_df.head())
    print("\nClassification report:")
    print(report_df)

    # ----- Matriz de confusão -----
    y_true = preds_df["y_true"]
    y_pred = preds_df["y_pred"]
    class_names = ["down_or_flat", "up"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.title("Matriz de Confusão - Direção BTC (teste)")
    plt.tight_layout()

    cm_path = FIGURES_DIR / "confusion_matrix_btc_direction.png"
    plt.savefig(cm_path, dpi=300)
    plt.show()
    print("Matriz de confusão salva em:", cm_path)

    # ----- Loss de treino -----
    plt.figure(figsize=(5, 4))
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o")
    plt.xlabel("Época")
    plt.ylabel("Loss de Treino")
    plt.title("Loss de Treino por Época - BTC Direction (FinBERT)")
    plt.grid(True)
    plt.tight_layout()

    loss_path = FIGURES_DIR / "train_loss_per_epoch.png"
    plt.savefig(loss_path, dpi=300)
    plt.show()
    print("Gráfico de loss salvo em:", loss_path)

    # ----- Accuracy de teste -----
    plt.figure(figsize=(5, 4))
    plt.plot(history_df["epoch"], history_df["test_acc"], marker="o", color="green")
    plt.xlabel("Época")
    plt.ylabel("Accuracy (Teste)")
    plt.title("Accuracy de Teste por Época - BTC Direction (FinBERT)")
    plt.grid(True)
    plt.tight_layout()

    acc_path = FIGURES_DIR / "test_accuracy_per_epoch.png"
    plt.savefig(acc_path, dpi=300)
    plt.show()
    print("Gráfico de accuracy salvo em:", acc_path)

    # ----- F1 de teste -----
    plt.figure(figsize=(5, 4))
    plt.plot(history_df["epoch"], history_df["test_f1_weighted"],
             marker="o", color="purple")
    plt.xlabel("Época")
    plt.ylabel("F1-weighted (Teste)")
    plt.title("F1-weighted de Teste por Época - BTC Direction (FinBERT)")
    plt.grid(True)
    plt.tight_layout()

    f1_path = FIGURES_DIR / "test_f1_per_epoch.png"
    plt.savefig(f1_path, dpi=300)
    plt.show()
    print("Gráfico de F1 salvo em:", f1_path)


if __name__ == "__main__":
    main()
