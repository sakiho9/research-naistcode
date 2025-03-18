# 必要に応じてパッケージをインストールしてください
# pip install pandas sentence-transformers scikit-learn numpy

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import numpy as np

# csvファイルを読み込む
logepisode = pd.read_csv('/Users/kuriusakiho/Documents/research/episode/data/logepisode (2).csv') # 適切なパスに変更してください

# エピソード本文の列に含まれる文字列を置換して削除する
logepisode['エピソード本文'] = logepisode['エピソード本文'].str.replace('_x000D_', '').str.replace('\n', '').str.replace('\r', '')
# 空白文字列を持つ行を除外する場合（オプション）
logepisode = logepisode[logepisode['エピソード本文'] != '']

# 8列目から47列目までを取得
logepisode = logepisode.iloc[:, 8:48]

# 'NaN'を実際のNaN値に置き換える
logepisode.replace('NaN', pd.NA, inplace=True)

# 無視したい列をリスト化
ignore_columns = ['タイトル', 'エピソード本文']

# 無視する列を除外したデータフレームを作成
df_themes = logepisode.drop(columns=ignore_columns)

# 行ごとのテーマの組み合わせを取得
logepisode['theme'] = df_themes.apply(lambda x: ','.join(x.dropna().index), axis=1)

# 空の組み合わせを「該当するテーマなし」と置き換える
logepisode['theme'] = logepisode['theme'].replace('', '該当するテーマなし')

# 「何に関するエピソードですか_」を削除
logepisode['theme'] = logepisode['theme'].str.replace('何に関するエピソードですか_', '')

# 組み合わせの頻度をカウント
combination_counts = logepisode['theme'].value_counts()

# 「該当するテーマなし」を除外
combination_counts = combination_counts[combination_counts.index != '該当するテーマなし']

# 上位9つの組み合わせを抽出
top_9_combinations = combination_counts.head(9)

# 上位9つの組み合わせをリスト化
top_9_combinations_list = top_9_combinations.index.tolist()

# テーマの組み合わせを「その他」と置き換え
def categorize_theme(theme_combination):
    if theme_combination in top_9_combinations_list:
        return theme_combination
    else:
        return 'その他'

# 新しい列にテーマを格納
logepisode['theme'] = logepisode['theme'].apply(categorize_theme)

# 「その他」を除外
filtered_logepisode = logepisode[logepisode['theme'] != 'その他']

# sentence-transformers のモデルをロード
model = SentenceTransformer('sonoisa/sentence-bert-base-ja-mean-tokens-v2')

# 各エピソード本文のベクトルを計算
def document_vector(doc):
    vectors = model.encode([doc])
    return vectors[0]

# 各エピソード本文のベクトルを計算
vectors = np.array([document_vector(doc) for doc in logepisode['エピソード本文']])

# t-SNEによる次元削減
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000, random_state=42)
tsne_vectors = tsne.fit_transform(vectors)

# t-SNEで得られた2次元ベクトルをDataFrameに追加
logepisode['tsne_x'] = tsne_vectors[:, 0]
logepisode['tsne_y'] = tsne_vectors[:, 1]

# CSVファイルに保存
logepisode.to_csv('/Users/kuriusakiho/Documents/research/episode/data/logepisode_sentence_transformers.csv', index=False) # 適切なパスに変更してください