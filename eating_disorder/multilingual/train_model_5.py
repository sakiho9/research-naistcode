import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 韓国語モデルのパス
MODEL_PATH = "/home/is/sakiho-k/research/notebooks/eating_disorder/multilingual/model/korean_model/checkpoint-4640"

# モデルとトークナイザーのロード
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 評価指標の計算
def compute_metrics(predictions, labels):
    preds = np.argmax(predictions, axis=-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# テストデータの前処理
def preprocess_test_data(df):
    texts = df["text"].tolist()
    labels = df["category"].tolist()
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": torch.tensor(labels)
    })
    return dataset

# PyTorch Dataset クラスの定義
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # バッチのデータをテンソルに変換して返す
        input_ids = torch.tensor(self.dataset["input_ids"][idx])
        attention_mask = torch.tensor(self.dataset["attention_mask"][idx])
        labels = torch.tensor(self.dataset["labels"][idx])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

# テストデータの評価
def evaluate_model_on_test_data(model, test_dataset, batch_size=16):
    model.eval()
    dataloader = DataLoader(test_dataset, batch_size=batch_size)
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", ncols=100):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()
            predictions.extend(logits)
            labels.extend(batch_labels.cpu().numpy())

    # 評価指標の計算
    metrics = compute_metrics(np.array(predictions), np.array(labels))
    return metrics

# 主処理
def main():
    # テストデータの読み込み
    test_diet_j = pd.read_csv('/home/is/sakiho-k/research/notebooks/eating_disorder/data/split/diet_j_test.csv')
    test_disorder_j = pd.read_csv('/home/is/sakiho-k/research/notebooks/eating_disorder/data/split/disorder_j_test.csv')

    # テストデータを結合
    test_data_j = pd.concat([test_diet_j, test_disorder_j], axis=0)

    # テストデータの前処理
    dataset = preprocess_test_data(test_data_j)

    # PyTorch Datasetに変換
    test_dataset = CustomDataset(dataset)

    # モデルの評価
    metrics = evaluate_model_on_test_data(model, test_dataset)

    # 結果の出力
    print("Evaluation Metrics on Japanese Test Data:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")

if __name__ == "__main__":
    main()