import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_graph(D, sim="cosine", threshold=0.5):
    #similarity criterion is cosine similarity
    #threshold is the value below which two documents are considered similar

    # Extract document texts
    docs = [(doc.doc_id, f"{doc.title or ''} {doc.abstract or ''}") for doc in D if doc.abstract]
    doc_ids, texts = zip(*docs)

    # Vectorize
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words='english')
    X = vectorizer.fit_transform(texts)

    if sim == "cosine":
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(X)

    # Dictionary to store similarities
    similarity_dict = {}
    for i, doc_id in enumerate(doc_ids):
        similar_docs = {
            doc_ids[j]: similarity_matrix[i, j]
            for j in range(len(doc_ids))
            if i != j and similarity_matrix[i, j] >= threshold  # Only store if above threshold
        }
        similarity_dict[doc_id] = similar_docs

    return similarity_dict

def undirected_page_rank(D ,sim="cosine", threshold=0.5, iterations=50):
    graph = build_graph(D, sim, threshold)

    # Initialize PageRank scores
    scores = {doc_id: 1.0 / len(graph) for doc_id in graph}
    N = len(D)
    p = 0.15

    for _ in range(iterations):  # Number of iterations
        new_scores = {}
        for doc_id in graph:
            # Calculate the score based on neighbors
            neighbor_score = sum(scores[neighbor]/len(graph[neighbor]) if len(graph[neighbor]) > 0 else 0 for neighbor in graph[doc_id])
            new_scores[doc_id] = p/N + (1-p) * neighbor_score

        scores = new_scores
    
    # Order scores
    sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}

    return sorted_scores

def improved_page_rank(doc_info ,sim="cosine", threshold=0.5, iterations=50):
    D = [doc_info[doc_id][0] for doc_id in doc_info]
    graph = build_graph(D, sim, threshold)

    # Initialize PageRank scores
    scores = {doc_id: 1.0 / len(graph) for doc_id in graph}
    N = len(D)
    p = 0.15

    for _ in range(iterations):  # Number of iterations
        new_scores = {}
        for doc_id in graph:
            neighbors = graph[doc_id]
            total_weight = sum(neighbors.values())
            if total_weight > 0:
                neighbor_score = sum(
                    scores[neighbor] * weight / total_weight
                    for neighbor, weight in neighbors.items()
                )
            else:
                neighbor_score = 0


            doc_prior_prob = doc_info[doc_id][1]  # original rank value of the document according to BM25F
            neighbor_prior_prob = sum(doc_info[neighbor][1] for neighbor in graph[doc_id])
            new_scores[doc_id] = (p * doc_prior_prob) / neighbor_prior_prob + (1 - p) * neighbor_score if neighbor_prior_prob != 0 else (1 - p) * neighbor_score


        scores = new_scores
    
    # Order scores
    sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}

    return sorted_scores


