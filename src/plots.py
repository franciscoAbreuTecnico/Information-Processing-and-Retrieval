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


## plots
import matplotlib.pyplot as plt

def plot_cluster_divergence(cluster_labels, kl_divergences, label_summaries):
    plt.figure(figsize=(12, 6))
    bars = plt.bar(cluster_labels, kl_divergences, color='skyblue')
    plt.title("KL Divergence of Clusters from Global Centroid")
    plt.xlabel("Cluster")
    plt.ylabel("KL Divergence")

    # Annotate each bar with top divergent terms
    for bar, label in zip(bars, label_summaries):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, label,
                 ha='center', va='bottom', fontsize=8, rotation=45)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
