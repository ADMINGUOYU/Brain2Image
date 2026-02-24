# Imports
import os
import torch
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN 
from sentence_transformers import SentenceTransformer

# =========================================================
# 💡 全局控制开关
# =========================================================
DO_CLUSTERING = False 
# =========================================================

# ---------------------------------------------------------
# 1. 设置 Hugging Face 环境变量
# ---------------------------------------------------------
os.environ["HF_HOME"] = "/mnt/afs/250010218/hf_cache" 
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ---------------------------------------------------------
# 2. 读取 EEG 数据及对应标签
# ---------------------------------------------------------
processed_dir = "datasets/processed"
os.makedirs(processed_dir, exist_ok=True) 

output_plot_dir = 'preprocess/label_refine'
os.makedirs(output_plot_dir, exist_ok=True)

things_images_df_path = os.path.join(processed_dir, "things_images_df.pkl")
things_images_df: pd.DataFrame = pd.read_pickle(things_images_df_path)

print("DataFrame Columns:", things_images_df.columns)
class_labels = things_images_df['class_label'].unique()
labels_list = class_labels.tolist()

# ---------------------------------------------------------
# 💡 定义要对比的模型字典 {模型HF路径 : 别名(用于命名)}
# ---------------------------------------------------------
models_to_run = {
    'all-MiniLM-L6-v2': 'minilm',
    'clip-ViT-B-32': 'clip'
}

# 开始遍历两个模型进行对比实验
for model_name, model_alias in models_to_run.items():
    print("\n" + "="*60)
    print(f"🚀 当前正在处理模型: {model_name}")
    print("="*60)

    # ---------------------------------------------------------
    # 3. 提取特征
    # ---------------------------------------------------------
    print(f"正在加载文本特征提取模型 ({model_name})...")
    # SBERT 原生支持加载 CLIP 模型进行 Text 编码
    text_model = SentenceTransformer(model_name) 

    print("正在计算 labels 的语义 embeddings...")
    embeddings = text_model.encode(labels_list, show_progress_bar=True) 

    # ---------------------------------------------------------
    # 4. HDBSCAN 密度聚类 (受 DO_CLUSTERING 控制)
    # ---------------------------------------------------------
    if DO_CLUSTERING:
        print("正在使用 HDBSCAN 进行语义密度聚类...")
        hdbscan_model = HDBSCAN(min_cluster_size=10, min_samples=3)
        cluster_ids = hdbscan_model.fit_predict(embeddings)

        n_clusters_ = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
        n_noise_ = list(cluster_ids).count(-1)

        print(f"[{model_alias}] 自动发现了 {n_clusters_} 个聚类簇。")
        print(f"[{model_alias}] 被标记为噪声 (未归类) 的语义数量: {n_noise_}")
    else:
        print("DO_CLUSTERING 为 False，跳过聚类阶段...")

    # ---------------------------------------------------------
    # 5. PCA 降维 (3 维，支持 2D 和 3D)
    # ---------------------------------------------------------
    print("正在进行 PCA 降维并生成可视化图表...")
    pca = PCA(n_components=3, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)

    # =========================================================
    # 5.1 生成 2D 可视化图表
    # =========================================================
    plt.figure(figsize=(12, 10))

    if DO_CLUSTERING:
        noise_mask = (cluster_ids == -1)
        clustered_mask = (cluster_ids != -1)

        plt.scatter(embeddings_pca[noise_mask, 0], embeddings_pca[noise_mask, 1], 
                    c='gray', s=20, alpha=0.3, edgecolors='none', label='Noise/Outliers')
        plt.scatter(embeddings_pca[clustered_mask, 0], embeddings_pca[clustered_mask, 1], 
                    c=cluster_ids[clustered_mask], cmap='tab20', s=40, alpha=0.8, edgecolors='none')

        plt.title(f'PCA 2D Projection ({model_name})\n{n_clusters_} Clusters Found', fontsize=16, fontweight='bold')
        plt.legend(loc='best')
        plot_filename_2d = f"semantic_clusters_pca_2d_{model_alias}.png"
    else:
        plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                    c='steelblue', s=30, alpha=0.7, edgecolors='none')
        plt.title(f'PCA 2D Projection of THINGS Concepts ({model_name})\n(No Clustering)', fontsize=16, fontweight='bold')
        plot_filename_2d = f"semantic_pca_2d_{model_alias}_no_clustering.png"

    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)

    for i, label in enumerate(labels_list):
        if i % 30 == 0:  
            plt.annotate(label, (embeddings_pca[i, 0], embeddings_pca[i, 1]), 
                         fontsize=9, alpha=0.8, xytext=(3, 3), textcoords='offset points')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plot_path_2d = os.path.join(output_plot_dir, plot_filename_2d)
    plt.savefig(plot_path_2d, dpi=300, bbox_inches='tight')
    print(f"2D 可视化图表已保存至 {plot_path_2d}")
    plt.close() # 释放内存避免多图重叠

    # =========================================================
    # 5.2 生成 3D 可视化图表
    # =========================================================
    fig = plt.figure(figsize=(12, 10))
    ax3d = fig.add_subplot(111, projection='3d')

    if DO_CLUSTERING:
        ax3d.scatter(embeddings_pca[noise_mask, 0], embeddings_pca[noise_mask, 1], embeddings_pca[noise_mask, 2],
                     c='gray', s=20, alpha=0.3, edgecolors='none', label='Noise/Outliers')
        ax3d.scatter(embeddings_pca[clustered_mask, 0], embeddings_pca[clustered_mask, 1], embeddings_pca[clustered_mask, 2],
                     c=cluster_ids[clustered_mask], cmap='tab20', s=40, alpha=0.8, edgecolors='none')

        ax3d.set_title(f'PCA 3D Projection ({model_name})\n{n_clusters_} Clusters Found', fontsize=16, fontweight='bold')
        ax3d.legend(loc='best')
        plot_filename_3d = f"semantic_clusters_pca_3d_{model_alias}.png"
    else:
        ax3d.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], embeddings_pca[:, 2],
                     c='steelblue', s=30, alpha=0.7, edgecolors='none')
        ax3d.set_title(f'PCA 3D Projection of THINGS Concepts ({model_name})', fontsize=16, fontweight='bold')
        plot_filename_3d = f"semantic_pca_3d_{model_alias}_no_clustering.png"

    ax3d.set_xlabel('Principal Component 1', fontsize=12)
    ax3d.set_ylabel('Principal Component 2', fontsize=12)
    ax3d.set_zlabel('Principal Component 3', fontsize=12)

    for i, label in enumerate(labels_list):
        if i % 30 == 0:  
            ax3d.text(embeddings_pca[i, 0], embeddings_pca[i, 1], embeddings_pca[i, 2], 
                      label, fontsize=9, alpha=0.8)

    plt.tight_layout()
    plot_path_3d = os.path.join(output_plot_dir, plot_filename_3d)
    plt.savefig(plot_path_3d, dpi=300, bbox_inches='tight')
    print(f"3D 可视化图表已保存至 {plot_path_3d}")
    plt.close() # 释放内存

    # ---------------------------------------------------------
    # 6 & 7. 后续处理 (仅在聚类时执行)
    # ---------------------------------------------------------
    if DO_CLUSTERING:
        label_to_cluster = {}
        for i, label in enumerate(labels_list):
            if cluster_ids[i] == -1:
                label_to_cluster[label] = "Noise_Outlier"
            else:
                label_to_cluster[label] = f"Cluster_{cluster_ids[i]}"

        # 动态添加列名，避免覆盖 (e.g., high_level_label_clip)
        col_name = f'high_level_label_{model_alias}'
        things_images_df[col_name] = things_images_df['class_label'].map(label_to_cluster)
        print(f"已成功将 High-level labels 映射至新列: {col_name}！")
        
        print("\n" + "-"*50)
        print(f"=== {model_alias.upper()} 模型聚类簇代表性标签 ===")
        print("-"*50)

        cluster_typical_labels = {}
        for c in range(n_clusters_):
            idx_in_cluster = np.where(cluster_ids == c)[0]
            cluster_embs = embeddings[idx_in_cluster]
            centroid = np.mean(cluster_embs, axis=0)
            distances = np.linalg.norm(cluster_embs - centroid, axis=1)
            
            typical_idx_relative = np.argmin(distances)
            typical_idx_absolute = idx_in_cluster[typical_idx_relative]
            typical_label = labels_list[typical_idx_absolute]
            
            closest_indices_relative = np.argsort(distances)[:5]
            closest_words = [labels_list[idx_in_cluster[i]] for i in closest_indices_relative]
            
            cluster_typical_labels[f"Cluster_{c}"] = typical_label
            print(f"Cluster {c} (包含 {len(idx_in_cluster)} 个词):")
            print(f"  🎯 Typical Label : {typical_label}")
            print(f"  📚 核心词汇群    : {', '.join(closest_words)}\n")

if DO_CLUSTERING:
    print("\n所有模型处理完毕。DataFrame 前 5 行展示:")
    display_cols = ['class_label', 'high_level_label_minilm', 'high_level_label_clip']
    print(things_images_df[display_cols].head())