import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from scipy.stats import entropy

def clustering(D, args=None):
    # Extract document texts
    docs = [(doc.doc_id, f"{doc.title or ''} {doc.abstract or ''}") for doc in D if doc.abstract]
    doc_ids, texts = zip(*docs)

    # Vectorize
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words='english')
    X = vectorizer.fit_transform(texts)

    # Fixed k = 10
    k = 10
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"k = {k}, silhouette score: {score:.4f}")

    # Organize output clusters
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

    # Cluster centroid
    cluster_centroid = np.asarray(X.mean(axis=0)).flatten()
    cluster_centroid += 1e-12
    cluster_centroid /= cluster_centroid.sum()

    if global_centroid is not None:
        global_vec = global_centroid + 1e-12
        global_vec /= global_vec.sum()
        kl_div = entropy(cluster_centroid, global_vec)
        divergence_terms = np.array(vectorizer.get_feature_names_out())[
            (cluster_centroid - global_vec).argsort()[-10:][::-1]
        ]
    else:
        kl_div = None
        divergence_terms = []

    distances = cosine_distances(X)
    medoid_index = distances.sum(axis=1).argmin()
    medoid_doc_id = ids[medoid_index]

    top_terms = np.array(vectorizer.get_feature_names_out())[cluster_centroid.argsort()[-10:][::-1]]

    return {
        'medoid': medoid_doc_id,
        'top_terms': list(top_terms),
        'divergent_terms': list(divergence_terms),
        'kl_divergence': float(kl_div) if kl_div else None
    }

def evaluate(X, labels, centroids):
    # Internal metrics
    silhouette = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X.toarray(), labels)
    db_score = davies_bouldin_score(X.toarray(), labels)

    # Variance decomposition
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

# HELPER METHODS
def compute_global_centroid(D):
    docs = [(doc.doc_id, f"{doc.title or ''} {doc.abstract or ''}") for doc in D if doc.abstract]
    _, texts = zip(*docs)

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    global_centroid = np.asarray(X.mean(axis=0)).flatten()

    return global_centroid, vectorizer