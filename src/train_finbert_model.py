# src/train_finbert_model.py

import os
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from src.paths import PROCESSED_DIR, MODELS_DIR, CSV_DIR

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted")

    return acc, f1, all_labels, all_preds


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ========== 1. Carregar série de preço ==========
    csv_path = PROCESSED_DIR / "btc_price_data.csv"
    df = pd.read_csv(csv_path)

    print("Colunas do arquivo:", df.columns.tolist())

    df = df.rename(columns={
        "Date": "date",
        "btc_price": "close"
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ========== 2. Retornos e target ==========
    df["return"] = df["close"].pct_change()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(float)
    df = df.dropna(subset=["return", "target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    print("Primeiras linhas de preço/return/target:")
    print(df.head())

    # ========== 3. Janelas de retornos ==========
    lookback = 30

    sequences = []
    labels = []

    for i in range(lookback - 1, len(df)):
        window_returns = df["return"].iloc[i - lookback + 1 : i + 1].values
        tokens = ["{:+.4f}".format(r) for r in window_returns]
        seq_text = " ".join(tokens)

        sequences.append(seq_text)
        labels.append(int(df["target"].iloc[i]))

    dates_for_examples = df["date"].iloc[lookback - 1 :].reset_index(drop=True)

    data = pd.DataFrame({
        "date": dates_for_examples,
        "sequence_text": sequences,
        "label": labels
    })

    print(data.head())

    # ========== 4. Split temporal 80/20 ==========
    n = len(data)
    split_idx = int(n * 0.8)

    train_df = data.iloc[:split_idx].reset_index(drop=True)
    test_df  = data.iloc[split_idx:].reset_index(drop=True)

    print("Tamanho treino:", len(train_df), " / teste:", len(test_df))
    print("Última data do treino:", train_df["date"].iloc[-1])
    print("Primeira data do teste:", test_df["date"].iloc[0])

    # ========== 5. Tokenizer FinBERT ==========
    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    max_length = 128

    def encode_texts(texts):
        return tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

    train_enc = encode_texts(train_df["sequence_text"])
    test_enc  = encode_texts(test_df["sequence_text"])

    train_labels = torch.tensor(train_df["label"].tolist())
    test_labels  = torch.tensor(test_df["label"].tolist())

    # ========== 6. Datasets / Dataloaders ==========
    train_dataset = TensorDataset(
        train_enc["input_ids"],
        train_enc["attention_mask"],
        train_labels
    )

    test_dataset = TensorDataset(
        test_enc["input_ids"],
        test_enc["attention_mask"],
        test_labels
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

    # ========== 7. Modelo FinBERT binário ==========
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=2,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # ========== 8. Treino ==========
    n_epochs = 3
    history = []

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        acc, f1, y_true, y_pred = evaluate(model, test_loader, device)

        print(f"Train loss: {train_loss:.4f} | "
              f"Test acc: {acc:.4f} | Test F1 (weighted): {f1:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_acc": acc,
            "test_f1_weighted": f1
        })

    # ========== 9. Salvar predições e reports ==========
    test_results_df = pd.DataFrame({
        "date": test_df["date"],
        "sequence_text": test_df["sequence_text"],
        "y_true": y_true,
        "y_pred": y_pred
    })
    preds_path = CSV_DIR / "finbert_btc_direction_predictions.csv"
    test_results_df.to_csv(preds_path, index=False)

    report_dict = classification_report(
        y_true, y_pred,
        target_names=["down_or_flat", "up"],
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = CSV_DIR / "finbert_btc_direction_classification_report.csv"
    report_df.to_csv(report_path)

    history_df = pd.DataFrame(history)
    history_path = CSV_DIR / "finbert_btc_direction_history.csv"
    history_df.to_csv(history_path, index=False)

    # ========== 10. Salvar modelo ==========
    model_dir = MODELS_DIR / "finbert_btc_direction_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print("\nArquivos salvos:")
    print("  Modelo  :", model_dir)
    print("  History :", history_path)
    print("  Preds   :", preds_path)
    print("  Report  :", report_path)


if __name__ == "__main__":
    main()
