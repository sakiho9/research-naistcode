import plotly.express as px
import pandas as pd

# Doc2Vec
# CSVファイルを読み込む
data = pd.read_csv('/Users/kuriusakiho/Documents/research/episode/data/logepisode_ex.csv')

# Plotlyの散布図を作成
fig = px.scatter(data, x='tsne_x', y='tsne_y', title='エピソードの分布',
                 hover_name='タイトル', color='theme',
                 custom_data=['エピソード本文'])

# カスタムマーカーの設定
fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=2, color='white')),
                  selector=dict(mode='markers'))

# テキストラベルの設定
fig.update_traces(textposition='top center')

# 背景色とタイトルの設定
fig.update_layout(
    plot_bgcolor='Lavender',
    title={
        'text': 'エピソードの分布',
        'font': {
            'size': 28,
            'color': 'darkorange'
        }
    }
)

# JSONファイルにデータを保存
fig_json = fig.to_json()
with open('scatter_plot_data.json', 'w') as f:
    f.write(fig_json)

# HTMLファイルの内容を生成
html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>Plotly Scatter Plot with Custom JS</title>
    <!-- Plotly.jsライブラリを読み込む -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        /* 中央揃えのためのスタイル */
        body {{
            display: flex;
            justify-content: center; /* 水平方向の中央揃え */
            align-items: center; /* 垂直方向の中央揃え */
            height: 100vh; /* ビューポートの高さに合わせる */
            margin: 0;
        }}
        #plotly-div {{
            width: 100%; /* 幅を100%に設定 */
            height: 80vh; /* ビューポートの80%の高さに設定 */
            max-width: 1200px; /* 必要に応じて最大幅を設定 */
        }}
        #episode-content {{
            margin-top: 20px;
            padding: 10px;
            border: 1px solid #ccc;
        }}
    </style>
</head>
<body>
    <!-- Plotlyの散布図を描画するための要素 -->
    <div id="plotly-div"></div>

    <!-- エピソード本文を表示するための要素 -->
    <div id="episode-content"></div>

    <!-- custom.jsを読み込む -->
    <script src="custom.js"></script>
    <script>
        // Plotlyの描画が終わった後に実行する関数
        function afterPlot() {{
            // Plotlyの図の要素を取得する
            var plot = document.getElementById('plotly-div');

            if (!plot) {{
                console.error("plotly-div が見つかりません");
                return;
            }}

            console.log("Plotlyの要素が見つかりました:", plot);

            // 点がクリックされたときの動作を定義する関数
            function handlePointClick(data) {{
                if (data && data.points && data.points.length > 0) {{
                    // ポイントに関連付けられたカスタムデータを取得する
                    var episodeContent = data.points[0].customdata[0];
                    console.log('クリックされたエピソード本文:', episodeContent);

                    // エピソード本文を表示する要素に設定する
                    var contentDiv = document.getElementById('episode-content');
                    if (contentDiv) {{
                        contentDiv.innerHTML = episodeContent;
                    }}
                }} else {{
                    console.error("クリックイベントデータが正しくありません:", data);
                }}
            }}

            // Plotlyのクリックイベントリスナーを設定する
            plot.on('plotly_click', handlePointClick);
        }}

        // JSONファイルを読み込んでPlotlyのグラフを描画
        fetch('scatter_plot_data.json')
            .then(response => response.json())
            .then(fig_json => {{
                // Plotlyの初期化と描画
                Plotly.react('plotly-div', fig_json.data, fig_json.layout)
                    .then(afterPlot)
                    .catch(function(error) {{
                        console.error('Plotlyの描画エラー:', error);
                    }});
            }})
            .catch(error => console.error('JSONファイルの読み込みエラー:', error));
    </script>
</body>
</html>
"""

# HTMLファイルに保存
with open('scatter_plot_with_content.html', 'w') as f:
    f.write(html_content)
