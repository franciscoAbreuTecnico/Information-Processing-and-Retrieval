import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import entropy
from scipy.special import rel_entr
from scipy.stats import entropy
from scipy.special import rel_entr
from sklearn.metrics.pairwise import cosine_distances
from preprocessing import preprocess_text

from plots import plot_results

def clustering(D, args=None):
    docs = [(doc.doc_id, f"{doc.title or ''} {doc.abstract or ''}") for doc in D if doc.abstract]
    doc_ids, texts = zip(*docs)

    vectorizer = TfidfVectorizer(
        max_df=0.8,
        min_df=5,
        stop_words='english',
        tokenizer=preprocess_text,
        preprocessor=None
    )

    X = vectorizer.fit_transform(texts)

    # Fixed k = 25
    k = 25
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"k = {k}, silhouette score: {score:.4f}")

    clusters = {}
    for doc_id, label in zip(doc_ids, labels):
        clusters.setdefault(label, []).append(doc_id)

    return [(label, cluster) for label, cluster in clusters.items()], vectorizer, X

def interpret(cluster, D, global_centroid=None, global_vectorizer=None, args=None):
    cluster_ids = set(cluster[1])
    docs = [(doc.doc_id, f"{doc.title or ''} {doc.abstract or ''}") for doc in D if doc.doc_id in cluster_ids]
    ids, texts = zip(*docs)

    vectorizer = global_vectorizer
    X = vectorizer.transform(texts)

    cluster_centroid = np.asarray(X.mean(axis=0)).flatten()
    cluster_centroid += 1e-12
    cluster_centroid /= cluster_centroid.sum()

    global_vec = global_centroid + 1e-12
    global_vec /= global_vec.sum()

    kl_div = entropy(cluster_centroid, global_vec)

    # Per-term KL contributions
    kl_terms = rel_entr(cluster_centroid, global_vec)
    top_kl_indices = kl_terms.argsort()[-10:][::-1]
    divergent_scores = kl_terms[top_kl_indices]
    divergent_terms = np.array(vectorizer.get_feature_names_out())[top_kl_indices]

    # Medoid doc
    distances = cosine_distances(X)
    medoid_index = distances.sum(axis=1).argmin()
    medoid_doc_id = ids[medoid_index]

    # Top terms by TF-IDF
    top_terms = np.array(vectorizer.get_feature_names_out())[cluster_centroid.argsort()[-10:][::-1]]

    return {
        'medoid': medoid_doc_id,
        'top_terms': list(top_terms),
        'divergent_terms': list(divergent_terms),
        'divergent_scores': list(divergent_scores),
        'kl_divergence': float(kl_div)
    }

def evaluate(X, labels, centroids):
    # Internal metrics
    silhouette = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X.toarray(), labels)
    db_score = davies_bouldin_score(X.toarray(), labels)

    sse = 0.0
    for i, centroid in enumerate(centroids):
        cluster_points = X[labels == i]
        sse += ((cluster_points - centroid) ** 2).sum()

    global_mean = X.mean(axis=0)
    tss = ((X - global_mean) ** 2).sum()
    ssb = tss - sse

    return {
        "k": len(set(labels)),
        "silhouette_score": silhouette,
        "calinski_harabasz_score": ch_score,
        "davies_bouldin_score": db_score,
        "SSE (intra-cluster)": sse,
        "TSS (total variance)": tss,
        "SSB (between clusters)": ssb,
        "SSB / TSS ratio": ssb / tss if tss else None
    }

def run_clustering_experiments(X, doc_ids, method_list, k_values):
    results = []

    for method in method_list:
        print(f"\n=== {method.upper()} Clustering ===")
        for k in k_values:
            if method == "kmeans":
                model = KMeans(n_clusters=k, random_state=42)
            elif method == "agglomerative":
                model = AgglomerativeClustering(n_clusters=k)
            else:
                continue

            labels = model.fit_predict(X.toarray() if method == "agglomerative" else X)

            silhouette = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X.toarray(), labels)
            db = davies_bouldin_score(X.toarray(), labels)

            print(f"k = {k:2d} | Silhouette: {silhouette:.4f} | Calinski-Harabasz: {ch:.2f} | Davies-Bouldin: {db:.4f}")
            results.append((method, k, silhouette, ch, db))

    plot_results(results)

def compute_global_centroid(D):
    docs = [(doc.doc_id, f"{doc.title or ''} {doc.abstract or ''}") for doc in D if doc.abstract]
    _, texts = zip(*docs)

    vectorizer = TfidfVectorizer(
        stop_words='english',
        tokenizer=preprocess_text,
        preprocessor=None
    )
    X = vectorizer.fit_transform(texts)
    global_centroid = np.asarray(X.mean(axis=0)).flatten()

    return global_centroid, vectorizer

def compute_global_centroid_represent_cluster(D, vectorizer):
    docs = [(doc.doc_id, f"{doc.title or ''} {doc.abstract or ''}") for doc in D if doc.abstract]
    _, texts = zip(*docs)

    X = vectorizer.transform(texts)
    global_centroid = np.asarray(X.mean(axis=0)).flatten()
    return global_centroid
