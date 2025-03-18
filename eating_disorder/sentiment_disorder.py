import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# トークナイザーとモデルをロード
tokenizer = AutoTokenizer.from_pretrained("Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime")
model = AutoModelForSequenceClassification.from_pretrained('Mizuiro-sakura/luke-japanese-large-sentiment-analysis-wrime')

# データフレームの読み込み
df = pd.read_csv('/home/is/sakiho-k/research/data/disorder_data.csv')

# NaNを空文字列に置き換え
df['text'] = df['text'].fillna('')

# 数値を文字列に変換
df['text'] = df['text'].astype(str)

# 感情ラベルの対応
emotions = ['joy、うれしい', 'sadness、悲しい', 'anticipation、期待', 'surprise、驚き', 
            'anger、怒り', 'fear、恐れ', 'disgust、嫌悪', 'trust、信頼']

# 感情スコアを格納するリスト
sentiment_scores = []

# 各テキストに対する感情分析
for text in df['text']:
    # トークン化とモデルへの入力準備
    max_seq_length = 512
    tokens = tokenizer(text,
                       truncation=True,
                       max_length=max_seq_length,
                       padding="max_length",
                       return_tensors="pt")  # Tensor形式で返す

    # モデルに入力
    with torch.no_grad():  # 推論時に勾配計算を無効化
        output = model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])

    # 最大スコアのインデックスを取得
    max_index = torch.argmax(output.logits, dim=1).item()

    # 感情ラベルを取得
    sentiment_scores.append(emotions[max_index])

# 新しい列を作成して感情スコアを追加
df['sentiment'] = sentiment_scores

# 結果をCSVファイルに保存（オプション）
df.to_csv('/home/is/sakiho-k/research/data/disorder_sent8.csv', index=False)