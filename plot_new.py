import torch
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = "pubmed"
weights_emb_path = f"checkpoints/{dataset}/2/similarity-weights.emb"
weights_tensor = torch.load(weights_emb_path).detach()
weights_np = weights_tensor.detach().cpu().numpy()
degrees_list_path = f"prt_lm/{dataset}/degrees_list.json"
with open(degrees_list_path, 'r') as f:
    degrees_list = json.load(f)

df = pd.DataFrame(weights_np, columns=["ORI", "IDR", "SAS", "SAR"])
df['Degree'] = degrees_list

# 列名
columns_names = ["ORI", "IDR", "SAS", "SAR"]

# 不同 degree 范围内的数据
df_0_10 = df[(df['Degree'] >= 0) & (df['Degree'] <= 10)]
df_10_50 = df[(df['Degree'] > 10) & (df['Degree'] <= 50)]
df_50_plus = df[df['Degree'] > 50]

# 节点总数
total_nodes = len(df)

# 为每个范围计算节点数目和占总节点数目的百分比
node_counts = [len(df_0_10), len(df_10_50), len(df_50_plus)]
node_percents = [(count / total_nodes) * 100 for count in node_counts]

# 计算平均值
mean_weights_0_10 = df_0_10[columns_names].mean()
mean_weights_10_50 = df_10_50[columns_names].mean()
mean_weights_50_plus = df_50_plus[columns_names].mean()

# 为每个范围绘制柱状图
for i, (mean_weights, degree_range, node_count, node_percent) in enumerate(zip([mean_weights_0_10, mean_weights_10_50, mean_weights_50_plus], 
                                                     ['0-10', '10-50', '50+'],
                                                     node_counts,
                                                     node_percents)):
    plt.figure(figsize=(8, 6))
    plt.bar(columns_names, mean_weights)
    plt.title(f'Average Weights Distribution for Degree {degree_range}\nNodes: {node_count} ({node_percent:.2f}%)')
    plt.xlabel('Weight Type')
    plt.ylabel('Average Weight')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'plots/{dataset}_weights_distribution_{degree_range}.png')
