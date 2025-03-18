import os
import pandas as pd
import torch
from tqdm import tqdm  # tqdm をインポート
from transformers import BertJapaneseTokenizer, AutoModelForSequenceClassification

# モデルとトークナイザーをロード
model_path = "/home/is/sakiho-k/research/notebooks/model/pretreated/checkpoint-22000"
tokenizer = BertJapaneseTokenizer.from_pretrained(model_path, do_lower_case=False)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 評価データをCSVから読み込む
data_path = "/home/is/sakiho-k/research/data/pretreated/remaining_blogs.csv"
df = pd.read_csv(data_path)

# テキストカラム内の欠損値や非文字列データを除外
df = df.dropna(subset=['text'])  # 欠損値を削除
df = df[df['text'].apply(lambda x: isinstance(x, str))]  # 文字列でない値を除外

# テキストとラベルを再取得
texts = df['text'].tolist()  # テキスト
true_labels = df['labels'].tolist()  # 実際のラベル

# モデルを評価モードに切り替え
model.eval()

# テキストをトークナイズ（TQDMを適用）
print("Tokenizing texts...")
inputs = tokenizer(
    texts,
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=128,
    verbose=True,  # tqdm を内部で利用
)

# 予測を実施（TQDMを適用）
print("Performing predictions...")
all_predictions = []
with torch.no_grad():
    for i in tqdm(range(0, len(texts), 32), desc="Processing batches"):  # バッチ単位で処理
        batch_texts = texts[i:i + 32]
        batch_inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        batch_outputs = model(**batch_inputs)
        batch_logits = batch_outputs.logits
        batch_predictions = torch.argmax(batch_logits, dim=-1).cpu().numpy()
        all_predictions.extend(batch_predictions)

# 予測結果をデータフレームに追加
df['Prediction'] = all_predictions

# TP, TN, FP, FN の判定列を作成
df['TP'] = ((df['labels'] == 1) & (df['Prediction'] == 1)).astype(int)
df['TN'] = ((df['labels'] == 0) & (df['Prediction'] == 0)).astype(int)
df['FP'] = ((df['labels'] == 0) & (df['Prediction'] == 1)).astype(int)
df['FN'] = ((df['labels'] == 1) & (df['Prediction'] == 0)).astype(int)

# 出力先ディレクトリが存在しない場合に作成
output_dir = "/home/is/sakiho-k/research/data/pretreated"
os.makedirs(output_dir, exist_ok=True)

# 結果をCSVファイルに保存
output_file = os.path.join(output_dir, "remaining_matrix.csv")
df.to_csv(output_file, index=False)

# ログ出力
print(f"Predictions and evaluations saved to {output_file}")