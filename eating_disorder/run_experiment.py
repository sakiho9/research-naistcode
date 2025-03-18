import os
import sys

# 使用するGPUを指定。PyTorchをimportする前に設定する必要があります
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# モジュールのパスを追加
sys.path.append('/home/is/sakiho-k/research')

# データを使って学習と評価を行う（ラベル1:1）
from src.eating_disorder.dataset import load_dataset_2class_classification, cross_validation

# データのロード
df_train = load_dataset_2class_classification('/home/is/sakiho-k/research/data/pretreated/train_data.csv')
df_test = load_dataset_2class_classification('/home/is/sakiho-k/research/data/pretreated/test_data.csv')

# 交差検証の実行
result = cross_validation(df_train, df_test)

# 結果の表示
average_accuracy = sum(d['eval_accuracy'] for d in result)/len(result)
average_macro_f1 = sum(d['eval_f1'] for d in result)/len(result)
average_recall = sum(d['eval_recall'] for d in result)/len(result)
average_precision = sum(d['eval_precision'] for d in result)/len(result)

print("Average accuracy:", average_accuracy)
print("Average Macro f1:", average_macro_f1)
print("Average Recall:", average_recall)
print("Average Precision:", average_precision)
