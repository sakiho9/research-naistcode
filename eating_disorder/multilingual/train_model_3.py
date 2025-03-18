import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os
from tqdm import tqdm
from transformers import EarlyStoppingCallback

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルとトークナイザーのロード
MODEL_NAME = "FacebookAI/xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 評価指標の計算
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# データの前処理
def preprocess_data(df):
    texts = df["text"].tolist()
    labels = df["category"].tolist()  # 'category'列をラベルとして使用
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels
    })
    return dataset

# モデルのトレーニング関数
def train_model(train_dataset, val_dataset, output_dir, epochs=100, early_stopping_patience=3):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="epoch",
        evaluation_strategy="epoch",  # ここでのみ設定
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        save_total_limit=1,
        load_best_model_at_end=True,  # モデルのパフォーマンスが最良のものを保存
        metric_for_best_model="accuracy",  # 最良モデルを選択する際に使用するメトリック
        greater_is_better=True,  # 精度を最大化する方向で選択
        logging_steps=500,  # ログを出力する間隔（任意）
    )

    # Early stoppingの設定
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)

    # Trainerの設定
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # 検証データセットを追加
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # 評価指標の関数を追加
        callbacks=[early_stopping_callback]  # EarlyStoppingCallbackを追加
    )

    trainer.train()
    return model

# クロスバリデーション関数
def cross_validate_and_evaluate(texts, labels, test_texts, test_labels, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    results = {"val": [], "test": []}
    best_val_metric = float("inf")
    best_model = None
    best_fold = None

    results_dir = "/home/is/sakiho-k/research/notebooks/eating_disorder/multilingual/results"
    os.makedirs(results_dir, exist_ok=True)
    results_file_path = os.path.join(results_dir, "all_results_j.txt")
    
    with open(results_file_path, "w", encoding="utf-8") as result_file:
        for fold, (train_idx, val_idx) in tqdm(enumerate(skf.split(texts, labels)), total=k, desc="Training Fold", ncols=100):
            train_texts, val_texts = [texts[i] for i in train_idx], [texts[i] for i in val_idx]
            train_labels, val_labels = [labels[i] for i in train_idx], [labels[i] for i in val_idx]
            
            train_dataset = preprocess_data(pd.DataFrame({"text": train_texts, "category": train_labels}))
            val_dataset = preprocess_data(pd.DataFrame({"text": val_texts, "category": val_labels}))
            test_dataset = preprocess_data(pd.DataFrame({"text": test_texts, "category": test_labels}))
            
            output_dir = f"/home/is/sakiho-k/research/notebooks/eating_disorder/multilingual/model_japan/model_fold_{fold}"
            fine_tuned_model, trainer = train_model(train_dataset, val_dataset, output_dir)
            
            val_metrics = trainer.evaluate(val_dataset)
            test_metrics = trainer.evaluate(test_dataset)
            
            results["val"].append(val_metrics)
            results["test"].append(test_metrics)
            
            result_file.write(f"--- Fold {fold} ---\n")
            result_file.write(f"Validation Metrics: {val_metrics}\n")
            result_file.write(f"Test Metrics: {test_metrics}\n\n")
            
            if val_metrics["eval_loss"] < best_val_metric:
                best_val_metric = val_metrics["eval_loss"]
                best_model = fine_tuned_model
                best_fold = fold
        
        if best_model:
            best_model.save_pretrained(f"/home/is/sakiho-k/research/notebooks/eating_disorder/multilingual/model_japan/best_model_fold_{best_fold}")
            tokenizer.save_pretrained(f"/home/is/sakiho-k/research/notebooks/eating_disorder/multilingual/model_japan/best_model_fold_{best_fold}")
            result_file.write(f"Best model saved from fold {best_fold} with validation loss {best_val_metric}\n")

        avg_val_metrics = {k: np.mean([fold[k] for fold in results["val"]]) for k in results["val"][0]}
        avg_test_metrics = {k: np.mean([fold[k] for fold in results["test"]]) for k in results["test"][0]}
        
        result_file.write(f"\n--- Final Results ---\n")
        result_file.write(f"Average Validation Metrics: {avg_val_metrics}\n")
        result_file.write(f"Average Test Metrics: {avg_test_metrics}\n")

# 主処理
def main():
    train_diet_j = pd.read_csv('/home/is/sakiho-k/research/notebooks/eating_disorder/data/split/diet_j_train.csv')
    test_diet_j = pd.read_csv('/home/is/sakiho-k/research/notebooks/eating_disorder/data/split/diet_j_test.csv')
    train_disorder_j = pd.read_csv('/home/is/sakiho-k/research/notebooks/eating_disorder/data/split/disorder_j_train.csv')
    test_disorder_j = pd.read_csv('/home/is/sakiho-k/research/notebooks/eating_disorder/data/split/disorder_j_test.csv')

    train_data_j = pd.concat([train_diet_j, train_disorder_j], axis=0)
    test_data_j = pd.concat([test_diet_j, test_disorder_j], axis=0)

    cross_validate_and_evaluate(
        train_data_j["text"].tolist(),
        train_data_j["category"].tolist(),
        test_data_j["text"].tolist(),
        test_data_j["category"].tolist()
    )

if __name__ == "__main__":
    main()