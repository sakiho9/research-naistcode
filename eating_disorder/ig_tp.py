import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import IntegratedGradients
import numpy as np
from tqdm import tqdm  # tqdmのインポート
import os

# CSVファイルの読み込み
df = pd.read_csv('/home/is/sakiho-k/research/data/pretreated/matrix.csv')

# データのフィルタリング
tp_df = df[df['TP'] == 1]
tn_df = df[df['TN'] == 1]
fp_df = df[df['FP'] == 1]
fn_df = df[df['FN'] == 1]

# モデルとトークナイザーの読み込み
model_name = '/home/is/sakiho-k/research/notebooks/model/pretreated/checkpoint-22000'
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# モデルの出力からlogitsだけを取り出す関数
def custom_forward(input_embeds, attention_mask):
    outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
    logits = outputs.logits
    return logits

# IntegratedGradientsのインスタンス化
ig = IntegratedGradients(custom_forward)

# テストデータフレームの全サンプルに対して寄与度を計算
def compute_attributions_for_all_samples(df):
    results = []  # 結果を保存するリスト
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing samples"):  # tqdmを使用
        sample_text = row['text']
        
        # 入力のトークン化
        inputs = tokenizer(sample_text, return_tensors='pt', max_length=256, truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # input_idsの埋め込み層を取得
        input_embeds = model.get_input_embeddings()(input_ids)
        input_embeds.requires_grad_(True)

        # 勾配計算
        try:
            attributions = ig.attribute(inputs=input_embeds, additional_forward_args=(attention_mask,), target=1)
            attributions = attributions.squeeze(0).tolist()

            # トークンと寄与度を取得
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
            attributions = [sum(attr) / len(attr) if isinstance(attr, list) else attr for attr in attributions]

            # トークンと寄与度をリストに追加
            for token, attribution in zip(tokens, attributions):
                results.append((token, attribution))
        
        except Exception as e:
            print(f"Error at index {idx}: {e}")

    return results

# 寄与度順にソートする関数
def sort_attributions(attributions):
    return sorted(attributions, key=lambda x: abs(x[1]), reverse=True)

# テストデータ全体で寄与度を計算してリスト化
attributions_list = compute_attributions_for_all_samples(tp_df)
sorted_attributions = sort_attributions(attributions_list)

# 出力先ディレクトリが存在しない場合に作成
output_dir = '/home/is/sakiho-k/research/results'
os.makedirs(output_dir, exist_ok=True)

# 結果をCSVファイルに保存（リストすべてを保存）
output_file = os.path.join(output_dir, 'all_attributions_tp.csv')
pd.DataFrame(attributions_list, columns=['Token', 'Attribution']).to_csv(output_file, index=False)

print(f"All attributions saved to {output_file}")