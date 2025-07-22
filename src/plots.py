from collections import Counter

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt

def plot_top_divergent_terms(cluster_id, terms, scores):
    plt.figure(figsize=(10, 5))
    plt.barh(terms[::-1], scores[::-1])
    plt.xlabel("KL Contribution")
    plt.title(f"Top Divergent Terms in Cluster {cluster_id}")
    plt.tight_layout()
    plt.show()

def plot_cluster_kl_divergence(cluster_labels, kl_divergences):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=kl_divergences, y=cluster_labels)
    plt.xlabel("KL Divergence from Global Distribution")
    plt.title("Cluster Divergence Overview")
    plt.tight_layout()
    plt.show()

def plot_wordcloud(terms, scores):
    wordcloud = WordCloud(width=800, height=400)
    frequencies = dict(zip(terms, scores))
    plt.imshow(wordcloud.generate_from_frequencies(frequencies), interpolation='bilinear')
    plt.axis("off")
    plt.show()

def plot_cluster_divergence(cluster_labels, kl_divergences, label_summaries):
    plt.figure(figsize=(12, 6))
    bars = plt.bar(cluster_labels, kl_divergences, color='skyblue')
    plt.title("KL Divergence of Clusters from Global Centroid")
    plt.xlabel("Cluster")
    plt.ylabel("KL Divergence")

    for bar, label in zip(bars, label_summaries):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, label,
                 ha='center', va='bottom', fontsize=8, rotation=45)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_tsne(X, labels, title="t-SNE of Clusters (Reduced TF-IDF)"):
    print("Reducing TF-IDF dimensions with TruncatedSVD...")
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_reduced = svd.fit_transform(X)

    print("Running t-SNE on reduced dimensions...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, metric='cosine')
    X_tsne = tsne.fit_transform(X_reduced)

    print("Plotting t-SNE results...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette='tab10', s=50, alpha=0.7)
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_results(results):
    import seaborn as sns
    import pandas as pd

    df = pd.DataFrame(results, columns=["Method", "k", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin"])
    sns.set(style="whitegrid")

    # Silhouette
    plt.figure()
    sns.lineplot(data=df, x="k", y="Silhouette", hue="Method", marker="o")
    plt.title("Silhouette Score by k")
    plt.show()

    # CH
    plt.figure()
    sns.lineplot(data=df, x="k", y="Calinski-Harabasz", hue="Method", marker="o")
    plt.title("Calinski-Harabasz Score by k")
    plt.show()

    # DB
    plt.figure()
    sns.lineplot(data=df, x="k", y="Davies-Bouldin", hue="Method", marker="o")
    plt.title("Davies-Bouldin Score by k")
    plt.show()
