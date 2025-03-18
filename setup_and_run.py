import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 必要なパッケージのインストール
install("sentence-transformers")
install("fugashi")
install("ipadic")
install("pandas")

from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os

# モデルのロード
model = SentenceTransformer('sonoisa/sentence-bert-base-ja-mean-tokens-v2')

# CSVファイルの読み込み
csv_file_path = 'data/logepisode.csv' # 適切なパスに変更してください
if not os.path.isfile(csv_file_path):
    raise FileNotFoundError(f"{csv_file_path} が見つかりません。正しいパスを指定してください。")

df = pd.read_csv(csv_file_path)
df['エピソード本文'] = df['エピソード本文'].str.replace('\r', '').str.replace('\n', '')

# エピソードのベクトル化
embeddings = model.encode(df['エピソード本文'].tolist())

def get_sim2(s1_index, s2_index):
    vec1 = embeddings[s1_index]
    vec2 = embeddings[s2_index]
    similarity = util.pytorch_cos_sim(vec1, vec2).item()
    return similarity

x = 6  # 比較したいエピソード番号に変更してください
print(f'比較したいエピソード: Episode {x}')
print(f'Episode: {df["エピソード本文"][x]}')
print('上位3つの類似エピソード')

similarity_scores = []
for i in range(len(df['エピソード本文'])):
    if i != x:
        similarity = get_sim2(x, i)
        similarity_scores.append((i, similarity))

# 類似度の高い順にソート
similarity_scores.sort(key=lambda x: x[1], reverse=True)

# 上位3つのエピソードを取得
top_3_episodes = similarity_scores[:3]

# 上位3つのエピソードを表示
for episode in top_3_episodes:
    episode_index = episode[0]
    similarity_score = episode[1]
    print(f"Episode {episode_index}: Similarity Score = {similarity_score}")
    print(df['エピソード本文'][episode_index])