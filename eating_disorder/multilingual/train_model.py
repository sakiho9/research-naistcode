import wandb
import pandas as pd
import numpy as np
import os
import warnings
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

warnings.filterwarnings('ignore')
os.environ['WANDB_SILENT'] = 'true'

def load_dataset_2class_classification(df):
    df_filtered = df[['text', 'category']].dropna()
    df_filtered = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
    texts = df_filtered['text'].values.tolist()
    labels = df_filtered['category'].values.tolist()
    return {'texts': texts, 'labels': labels}

def tokenize_function(data, tokenizer, max_len):
    return tokenizer(data['texts'], padding='max_length', truncation=True, max_length=max_len)

def preprocess_for_Trainer(dataset, tokenizer, max_len):
    data = Dataset.from_dict(dataset)
    data = data.map(
        tokenize_function,
        batched=True,
        fn_kwargs={'tokenizer': tokenizer, 'max_len': max_len},
        desc="Tokenizing Dataset"
    )
    return data

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def train_model_with_cv(train_data, test_data, learning_rate, num_epochs=100, batch_size=16, max_len=128, run_name='model', num_folds=5, is_korean=False, tokenizer=None, model=None):
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
    model_name = "FacebookAI/xlm-roberta-base"
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    num_labels = 2
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []
    test_fold_results = []  # テストデータ評価結果を保存
    korean_japanese_test_results = []  # 韓国語モデルで日本語テストデータを評価した結果
    japanese_korean_test_results = []  # 日本語モデルで韓国語テストデータを評価した結果
    japanese_fold_results = []  # 日本語モデルのfold結果を格納

    # ファイルオープン
    with open("test_results.txt", "w") as file:
        file.write("Test Results for Each Fold\n")
        file.write("="*50 + "\n")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_data['texts'], train_data['labels'])):
            print(f"\nTraining fold {fold + 1}/{num_folds}...")
            # トレーニングとバリデーションデータの準備
            train_dataset = preprocess_for_Trainer({'texts': [train_data['texts'][i] for i in train_idx],
                                                    'labels': [train_data['labels'][i] for i in train_idx]},
                                                    tokenizer, max_len)
            val_dataset = preprocess_for_Trainer({'texts': [train_data['texts'][i] for i in val_idx],
                                                  'labels': [train_data['labels'][i] for i in val_idx]},
                                                  tokenizer, max_len)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            
            # トレーニング引数の設定
            training_args = TrainingArguments(
                output_dir='./outputs/final_model_korean' if is_korean else './outputs/final_model',
                logging_dir='./logs',
                report_to='wandb',
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                logging_strategy="epoch",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                save_total_limit=1,
                logging_first_step=True
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
            # モデルのトレーニング
            trainer.train()
            # バリデーションセットで評価
            eval_result = trainer.evaluate(eval_dataset=val_dataset)
            fold_results.append(eval_result)
            japanese_fold_results.append(eval_result)  # 日本語モデルのfold結果も追加

            # テストデータで評価
            print(f"Evaluating fold {fold + 1}/{num_folds} on test data...")
            test_dataset = preprocess_for_Trainer(test_data, tokenizer, max_len)
            test_result = trainer.evaluate(eval_dataset=test_dataset)
            test_fold_results.append(test_result)  # 各foldのテスト評価結果を追加

            # 日本語でのテスト評価（韓国語モデル）
            print(f"Evaluating fold {fold + 1}/{num_folds} on Japanese test data...")
            test_result_japanese = evaluate_model(trainer, test_data_j, tokenizer, max_len, dataset_name="Japanese", model_name="Korean_Model_on_Japanese_Test_Data")
            korean_japanese_test_results.append(test_result_japanese)

            # 韓国語でのテスト評価（日本語モデル）
            print(f"Evaluating fold {fold + 1}/{num_folds} on Korean test data...")
            test_result_korean = evaluate_model(trainer, test_data_k, tokenizer, max_len, dataset_name="Korean", model_name="Japanese_Model_on_Korean_Test_Data")
            japanese_korean_test_results.append(test_result_korean)

            # 各foldのテストデータ評価結果を書き込む
            file.write(f"Fold {fold + 1} Test Results:\n")
            for key, value in test_result.items():
                file.write(f"  {key}: {value:.4f}\n")
            file.write("-"*50 + "\n")

    # 各foldごとの評価結果の平均を計算
    avg_results = {
        "accuracy": np.mean([result["eval_accuracy"] for result in fold_results]),
        "precision": np.mean([result["eval_precision"] for result in fold_results]),
        "recall": np.mean([result["eval_recall"] for result in fold_results]),
        "f1": np.mean([result["eval_f1"] for result in fold_results]),
    }

    # テストデータの平均評価結果を計算
    avg_test_results = {
        "accuracy": np.mean([result["eval_accuracy"] for result in test_fold_results]),
        "precision": np.mean([result["eval_precision"] for result in test_fold_results]),
        "recall": np.mean([result["eval_recall"] for result in test_fold_results]),
        "f1": np.mean([result["eval_f1"] for result in test_fold_results]),
    }

    avg_japanese_korean_test_results = {
        "accuracy": np.mean([result["eval_accuracy"] for result in korean_japanese_test_results]),
        "precision": np.mean([result["eval_precision"] for result in korean_japanese_test_results]),
        "recall": np.mean([result["eval_recall"] for result in korean_japanese_test_results]),
        "f1": np.mean([result["eval_f1"] for result in korean_japanese_test_results]),
    }

    avg_korean_japanese_test_results = {
        "accuracy": np.mean([result["eval_accuracy"] for result in korean_japanese_test_results]),
        "precision": np.mean([result["eval_precision"] for result in korean_japanese_test_results]),
        "recall": np.mean([result["eval_recall"] for result in korean_japanese_test_results]),
        "f1": np.mean([result["eval_f1"] for result in korean_japanese_test_results]),
    }

    write_detailed_results_to_file(
        "model_performance_results.txt", 
        japanese_fold_results, 
        korean_fold_results, 
        japanese_avg_results, 
        korean_avg_results, 
        japanese_test_eval, 
        korean_test_eval, 
        japanese_korean_test_eval,  # 日本語モデルで韓国語データを評価した結果
        korean_japanese_test_eval   # 韓国語モデルで日本語データを評価した結果
    )

    return avg_results, avg_test_results, avg_japanese_korean_test_results, avg_korean_japanese_test_results

# 日本語モデルの再ファインチューニング（全データ使用）
def fine_tune_japanese_model_all_data(train_data_j, tokenizer, model_name="FacebookAI/xlm-roberta-base", max_len=128, num_epochs=100, learning_rate=2e-5, batch_size=16):
    # FacebookAI/xlm-roberta-baseモデルを最初からロード
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # データの前処理
    train_dataset = preprocess_for_Trainer(train_data_j, tokenizer, max_len)

    # トレーニング引数の設定
    training_args = TrainingArguments(
        output_dir='./outputs/final_model_japanese',
        logging_dir='./logs',
        report_to='wandb',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
        logging_first_step=True
    )

    # Trainerの初期化
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # モデルの訓練
    trainer.train()

    # モデルを保存
    model_save_path = "japanese_finetuned_model_all_data"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    trainer.save_model(model_save_path)

    return model_save_path


def evaluate_model(trainer, test_data, tokenizer, max_len, dataset_name, model_name):
    test_dataset = preprocess_for_Trainer(test_data, tokenizer, max_len)
    eval_result = trainer.evaluate(eval_dataset=test_dataset)
    
    # ファイルに結果を書き込む
    with open('evaluation_results.txt', 'a') as f:
        f.write(f"\n[{model_name}] on {dataset_name} Test Dataset:\n")
        for key, value in eval_result.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\n" + "-"*50 + "\n")
    
    return eval_result

# 評価結果をテキストファイルに書き込む関数
def write_detailed_results_to_file(filename, japanese_fold_results, korean_fold_results, 
                                   japanese_avg_results, korean_avg_results, 
                                   japanese_test_eval, korean_test_eval, 
                                   japanese_korean_test_eval, korean_japanese_test_eval):
    """
    Writes detailed results of model evaluation to a specified file.

    Args:
        filename (str): The name of the file to write the results to.
        japanese_fold_results (list): List of evaluation results for Japanese model on validation data for each fold.
        korean_fold_results (list): List of evaluation results for Korean model on validation data for each fold.
        japanese_avg_results (dict): Average evaluation results for Japanese model on validation data.
        korean_avg_results (dict): Average evaluation results for Korean model on validation data.
        japanese_test_eval (dict): Evaluation results for Japanese model on test data.
        korean_test_eval (dict): Evaluation results for Korean model on test data.
        japanese_korean_test_eval (dict): Evaluation results for Japanese model when tested on Korean data.
        korean_japanese_test_eval (dict): Evaluation results for Korean model when tested on Japanese data.
    """
    with open(filename, 'w') as f:
        f.write("Detailed Results for Cross-Validation and Test Data\n")
        f.write("="*50 + "\n")

        # Japanese Model Results
        f.write("Japanese Model:\n")
        f.write("  Validation Results by Fold:\n")
        for i, fold_result in enumerate(japanese_fold_results):
            f.write(f"    Fold {i+1}:\n")
            for key, value in fold_result.items():
                f.write(f"      {key}: {value:.4f}\n")
        
        f.write("  Test Results by Fold (Japanese Model):\n")
        for i, test_result in enumerate(japanese_test_eval):
            f.write(f"    Fold {i+1} Japanese Test Results:\n")
            for key, value in test_result.items():
                f.write(f"      {key}: {value:.4f}\n")
        
        f.write("  Test Results by Fold (Korean Test Data for Japanese Model):\n")
        for i, test_result in enumerate(japanese_korean_test_eval):
            f.write(f"    Fold {i+1} Korean Test Results for Japanese Model:\n")
            for key, value in test_result.items():
                f.write(f"      {key}: {value:.4f}\n")
        
        f.write("  Average Validation Results (Japanese Model):\n")
        for key, value in japanese_avg_results.items():
            f.write(f"    {key}: {value:.4f}\n")
        
        f.write("  Japanese Model on Test Data (Average):\n")
        for key, value in japanese_test_eval.items():
            f.write(f"    {key}: {value:.4f}\n")
        
        f.write("  Japanese Model on Korean Test Data (Average):\n")
        for key, value in japanese_korean_test_eval.items():
            f.write(f"    {key}: {value:.4f}\n")
        f.write("\n")

        # Korean Model Results
        f.write("Korean Model:\n")
        f.write("  Validation Results by Fold:\n")
        for i, fold_result in enumerate(korean_fold_results):
            f.write(f"    Fold {i+1}:\n")
            for key, value in fold_result.items():
                f.write(f"      {key}: {value:.4f}\n")
        
        f.write("  Test Results by Fold (Korean Model):\n")
        for i, test_result in enumerate(korean_test_eval):
            f.write(f"    Fold {i+1} Korean Test Results:\n")
            for key, value in test_result.items():
                f.write(f"      {key}: {value:.4f}\n")
        
        f.write("  Test Results by Fold (Japanese Test Data for Korean Model):\n")
        for i, test_result in enumerate(korean_japanese_test_eval):
            f.write(f"    Fold {i+1} Japanese Test Results for Korean Model:\n")
            for key, value in test_result.items():
                f.write(f"      {key}: {value:.4f}\n")
        
        f.write("  Average Validation Results (Korean Model):\n")
        for key, value in korean_avg_results.items():
            f.write(f"    {key}: {value:.4f}\n")
        
        f.write("  Korean Model on Test Data (Average):\n")
        for key, value in korean_test_eval.items():
            f.write(f"    {key}: {value:.4f}\n")
        
        f.write("  Korean Model on Japanese Test Data (Average):\n")
        for key, value in korean_japanese_test_eval.items():
            f.write(f"    {key}: {value:.4f}\n")
        f.write("\n")

    print(f"Results written to {filename}")

if __name__ == "__main__":
    # 日本語データセットのロード
    train_diet_j = pd.read_csv('/home/is/sakiho-k/research/notebooks/eating_disorder/data/split/diet_j_train.csv')
    test_diet_j = pd.read_csv('/home/is/sakiho-k/research/notebooks/eating_disorder/data/split/diet_j_test.csv')
    train_disorder_j = pd.read_csv('/home/is/sakiho-k/research/notebooks/eating_disorder/data/split/disorder_j_train.csv')
    test_disorder_j = pd.read_csv('/home/is/sakiho-k/research/notebooks/eating_disorder/data/split/disorder_j_test.csv')

    train_data_j = load_dataset_2class_classification(pd.concat([train_diet_j, train_disorder_j], axis=0))
    test_data_j = load_dataset_2class_classification(pd.concat([test_diet_j, test_disorder_j], axis=0))

    # 韓国語データセットのロード
    train_diet_k = pd.read_csv('/home/is/sakiho-k/research/notebooks/eating_disorder/data/split/diet_k_train.csv')
    test_diet_k = pd.read_csv('/home/is/sakiho-k/research/notebooks/eating_disorder/data/split/diet_k_test.csv')
    train_disorder_k = pd.read_csv('/home/is/sakiho-k/research/notebooks/eating_disorder/data/split/disorder_k_train.csv')
    test_disorder_k = pd.read_csv('/home/is/sakiho-k/research/notebooks/eating_disorder/data/split/disorder_k_test.csv')

    train_data_k = load_dataset_2class_classification(pd.concat([train_diet_k, train_disorder_k], axis=0))
    test_data_k = load_dataset_2class_classification(pd.concat([test_diet_k, test_disorder_k], axis=0))

    # 日本語モデルの5-foldクロスバリデーション
    japanese_trainers, japanese_fold_results, japanese_avg_results = train_model_with_cv(
    train_data=train_data_j, test_data=test_data_j, learning_rate=2e-5, num_epochs=100, batch_size=16, max_len=128, run_name='japanese_model', is_korean=False)
    # トークナイザーのロード
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")

    # 日本語モデルの評価
    best_japanese_fold_idx = np.argmax([result["eval_accuracy"] for result in japanese_fold_results])
    japanese_test_eval = evaluate_model(japanese_trainers[best_japanese_fold_idx], test_data_j, tokenizer, max_len=128, dataset_name="Japanese", model_name="Best_Japanese_Model")

    # 日本語モデルでのファインチューニング（全データ使用）
    japanese_model_save_path = fine_tune_japanese_model_all_data(train_data_j, tokenizer)
    print(f"Japanese model fine-tuned and saved at {japanese_model_save_path}")

    # 韓国語モデルの5-foldクロスバリデーション（日本語でファインチューニングされたモデルを使用）
    def train_korean_model_with_new_japanese_finetuned_model(train_data_k, test_data_k, japanese_model_path, tokenizer, num_epochs=100, batch_size=16, max_len=128):
        # 新たにファインチューニングされた日本語モデルをロード
        model = AutoModelForSequenceClassification.from_pretrained(japanese_model_path, num_labels=2)

        # ベースモデル部分を凍結
        for param in model.base_model.parameters():
            param.requires_grad = False

        # 分類層（head）だけを学習可能にする
        for param in model.classifier.parameters():
            param.requires_grad = True
 
        # 韓国語データでクロスバリデーション
        korean_trainers, korean_fold_results, korean_avg_results = train_model_with_cv(
            train_data=train_data_k,
            eval_data=test_data_k,
            learning_rate=2e-5,
            num_epochs=num_epochs,
            batch_size=batch_size,
            max_len=max_len,
            run_name='korean_model_with_finetuned_japanese',
            is_korean=True,
            tokenizer=tokenizer,  # 日本語モデルのトークナイザーを渡す
            model=model           # 日本語モデルを渡す
        )
        return korean_trainers, korean_fold_results, korean_avg_results

    # 韓国語モデルのクロスバリデーション（ファインチューニングされた日本語モデルを使用）
    korean_trainers, korean_fold_results, korean_avg_results = train_korean_model_with_new_japanese_finetuned_model(
        train_data_k, test_data_k, japanese_model_save_path, tokenizer
    )

    # 韓国語モデルの評価
    best_korean_fold_idx = np.argmax([result["eval_accuracy"] for result in korean_fold_results])
    korean_test_eval = evaluate_model(korean_trainers[best_korean_fold_idx], test_data_k, tokenizer, max_len=128, dataset_name="Korean", model_name="Best_Korean_Model")

    # 日本語モデルで韓国語データを評価
    japanese_korean_test_eval = evaluate_model(japanese_trainers[best_japanese_fold_idx], test_data_k, tokenizer, max_len=128, dataset_name="Korean", model_name="Japanese_Model_on_Korean_Test_Data")

    # 韓国語モデルで日本語データを評価
    korean_japanese_test_eval = evaluate_model(korean_trainers[best_korean_fold_idx], test_data_j, tokenizer, max_len=128, dataset_name="Japanese", model_name="Korean_Model_on_Japanese_Test_Data")

    # 評価結果をテキストファイルに保存
    result_filepath = "model_performance_results.txt"
    write_detailed_results_to_file(
        result_filepath, 
        japanese_fold_results, korean_fold_results, 
        japanese_avg_results, korean_avg_results, 
        japanese_test_eval, korean_test_eval, 
        japanese_korean_test_eval, korean_japanese_test_eval
    )

    print(f"Evaluation results saved to {result_filepath}")

