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
def preprocess_data(df, tokenizer):
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
def train_model(train_dataset, eval_dataset, output_dir, epochs=100):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="epoch",  # モデル保存はエポックごと
        evaluation_strategy="epoch",  # バリデーションデータでエポックごとに評価
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        save_total_limit=1,
        load_best_model_at_end=True,  # 最良モデルの読み込みを有効化
        metric_for_best_model="eval_loss",  # 最良モデルの評価指標
        greater_is_better=False,  # 最小化する評価指標（損失）
    )
    
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)  # 早期停止を設定
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # バリデーションデータの設定
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback]  # 早期停止をコールバックに追加
    )
    
    trainer.train()
    return model, trainer

# クロスバリデーション関数
def cross_validate_and_evaluate(tokenizer, texts, labels, test_texts, test_labels, num_folds=5):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    results = {"val": [], "test": []}

    # 結果保存ディレクトリ
    results_dir = "/home/is/sakiho-k/research/notebooks/eating_disorder/multilingual/results"
    os.makedirs(results_dir, exist_ok=True)
    results_file_path = os.path.join(results_dir, "all_results.txt")

    with open(results_file_path, "w", encoding="utf-8") as result_file:
        for fold, (train_idx, val_idx) in tqdm(enumerate(skf.split(texts, labels)), total=num_folds, desc="Training Fold", ncols=100):
            # 各foldでデータを分割
            train_texts, val_texts = [texts[i] for i in train_idx], [texts[i] for i in val_idx]
            train_labels, val_labels = [labels[i] for i in train_idx], [labels[i] for i in val_idx]

            # データセットの作成
            train_dataset = preprocess_data(pd.DataFrame({"text": train_texts, "category": train_labels}), tokenizer)
            val_dataset = preprocess_data(pd.DataFrame({"text": val_texts, "category": val_labels}), tokenizer)
            test_dataset = preprocess_data(pd.DataFrame({"text": test_texts, "category": test_labels}), tokenizer)

            # クロスバリデーション関数の中でモデルをロード
            model_dir = "/home/is/sakiho-k/research/notebooks/eating_disorder/multilingual/model/korean_model"

            # モデルディレクトリの存在確認
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"モデル保存ディレクトリが存在しません: {model_dir}")

            model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)
            model.to(device)

            # Trainerのセットアップ
            training_args = TrainingArguments(
                output_dir=f"/home/is/sakiho-k/research/notebooks/eating_disorder/multilingual/model_korean_japan/model_fold_{fold}",
                save_strategy="epoch",
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                num_train_epochs=5,
                weight_decay=0.01,
                logging_dir=f"/home/is/sakiho-k/research/notebooks/eating_disorder/multilingual/model_korean_japan/model_fold_{fold}/logs",
                save_total_limit=1,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
            )
            early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                callbacks=[early_stopping_callback],
            )

            # 学習と評価
            trainer.train()
            val_metrics = trainer.evaluate(val_dataset)
            test_metrics = trainer.evaluate(test_dataset)

            # 各foldの結果をログに書き込む
            result_file.write(f"--- Fold {fold} ---\n")
            result_file.write(f"Validation Metrics: {val_metrics}\n")
            result_file.write(f"Test Metrics: {test_metrics}\n\n")

            # 結果を保存
            results["val"].append(val_metrics)
            results["test"].append(test_metrics)

        # 平均結果を計算
        avg_val_metrics = {k: np.mean([fold[k] for fold in results["val"]]) for k in results["val"][0]}
        avg_test_metrics = {k: np.mean([fold[k] for fold in results["test"]]) for k in results["test"][0]}

        # 平均結果をログに書き込む
        result_file.write("\n--- Final Results ---\n")
        result_file.write("Average Validation Metrics:\n")
        for k, v in avg_val_metrics.items():
            result_file.write(f"{k}: {v:.4f}\n")
        result_file.write("\nAverage Test Metrics:\n")
        for k, v in avg_test_metrics.items():
            result_file.write(f"{k}: {v:.4f}\n")

    return avg_test_metrics

# 主処理
def main():
    # 日本語データセットのロード
    train_diet_j = pd.read_csv('/home/is/sakiho-k/research/notebooks/eating_disorder/data/split/diet_j_train.csv')
    test_diet_j = pd.read_csv('/home/is/sakiho-k/research/notebooks/eating_disorder/data/split/diet_j_test.csv')
    train_disorder_j = pd.read_csv('/home/is/sakiho-k/research/notebooks/eating_disorder/data/split/disorder_j_train.csv')
    test_disorder_j = pd.read_csv('/home/is/sakiho-k/research/notebooks/eating_disorder/data/split/disorder_j_test.csv')

    train_data_j = pd.concat([train_diet_j, train_disorder_j], axis=0)
    test_data_j = pd.concat([test_diet_j, test_disorder_j], axis=0)

    # 韓国語データセットのロード
    train_diet_k = pd.read_csv('/home/is/sakiho-k/research/notebooks/eating_disorder/data/split/diet_k_train.csv')
    train_disorder_k = pd.read_csv('/home/is/sakiho-k/research/notebooks/eating_disorder/data/split/disorder_k_train.csv')
    train_data_k = pd.concat([train_diet_k, train_disorder_k], axis=0)

    # 韓国語の訓練データと検証データを分ける
    korean_train_data = train_data_k.sample(frac=0.8, random_state=42)  # 訓練データ（80%）
    korean_val_data = train_data_k.drop(korean_train_data.index)  # 検証データ（20%）

    # 韓国語モデルのファインチューニング
    korean_train_dataset = preprocess_data(korean_train_data, tokenizer)
    korean_val_dataset = preprocess_data(korean_val_data, tokenizer)

    # 韓国語モデルの保存ディレクトリ
    korean_model_dir = "/home/is/sakiho-k/research/notebooks/eating_disorder/multilingual/model/korean_model"

    # ディレクトリが存在しない場合は作成
    os.makedirs(korean_model_dir, exist_ok=True)
    # 韓国語モデルのファインチューニング
    korean_model, _ = train_model(korean_train_dataset, korean_val_dataset, korean_model_dir)
    # ファインチューニング済みの韓国語モデルを保存
    korean_model.save_pretrained(korean_model_dir)

    # 日本語データでクロスバリデーションと評価
    cross_validate_and_evaluate(
        tokenizer=tokenizer,
        texts=train_data_j["text"].tolist(),
        labels=train_data_j["category"].tolist(),
        test_texts=test_data_j["text"].tolist(),
        test_labels=test_data_j["category"].tolist()
    )

if __name__ == "__main__":
    main()