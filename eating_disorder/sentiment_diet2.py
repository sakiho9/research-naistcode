import re
import pandas as pd
from transformers import pipeline
import torch

def split_into_sentences(text):
    sentences = re.split(r'(?<=[。！？])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def split_and_merge(text):
    sentences = split_into_sentences(text)
    final_sentences = []
    for sentence in sentences:
        lines = sentence.split('\n')
        for line in lines:
            line = line.strip()
            if line:
                final_sentences.append(line)
    return final_sentences

def analyze_sentiment(sentences, sentiment_classifier):
    sentiments = []
    device = sentiment_classifier.device  # パイプラインのデバイスを取得

    for sentence in sentences:
        # トークナイザーによるトークン化
        inputs = sentiment_classifier.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        # モデルのデバイスに入力を移動
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # モデルに入力
        outputs = sentiment_classifier.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)

        # 各ラベルのスコアを取得
        scores = {sentiment_classifier.model.config.id2label[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        
        positive_score = scores.get('positive', 0)
        negative_score = scores.get('negative', 0)
        sentiment_score = positive_score - negative_score
        sentiments.append(sentiment_score)
    return sentiments

def calculate_average_score(sentiments):
    return sum(sentiments) / len(sentiments) if sentiments else 0

# モデルとトークナイザーの準備
model_name = 'lxyuan/distilbert-base-multilingual-cased-sentiments-student'  # モデル名を変更
device = 0 if torch.cuda.is_available() else -1

sentiment_classifier = pipeline(
    model=model_name,
    tokenizer=model_name,
    return_all_scores=True,
    device=device
)

# データフレームの読み込み
df = pd.read_csv('/home/is/sakiho-k/research/data/diet_data.csv')

# NaNを空文字列に置き換え
df['text'] = df['text'].fillna('')

# 数値を文字列に変換
df['text'] = df['text'].astype(str)

# 感情分析を行い、スコアをデータフレームに追加
def apply_sentiment_analysis(text):
    sentences = split_and_merge(text)
    sentiments = analyze_sentiment(sentences, sentiment_classifier)
    average_score = calculate_average_score(sentiments)
    return average_score

df['sentiment_score'] = df['text'].apply(apply_sentiment_analysis)

# 結果をCSVファイルに保存
df.to_csv('/home/is/sakiho-k/research/data/diet_score2.csv', index=False)