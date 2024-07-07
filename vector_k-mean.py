import os
import fitz
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

def read_pdf(file_path):
    document = fitz.open(file_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def read_pdfs_from_directory(directory):
    abstracts = []
    for f in os.listdir(directory):
        if f.endswith(".pdf"):
            abstracts.append(read_pdf(os.path.join(directory, f)))
    if len(abstracts) == 0:
        raise ValueError("No PDF files found in the specified directory.")
    return abstracts

def fit_tfidf_vectorizer(abstracts):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_vectors = vectorizer.fit_transform(abstracts)
    return tfidf_vectors, tfidf_vectors.toarray()  # Return both sparse and dense versions

def determine_optimal_clusters(tfidf_vectors, max_clusters):
    sse = []
    silhouette_scores = []
    davies_bouldin_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(tfidf_vectors)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(tfidf_vectors, kmeans.labels_))
        davies_bouldin_scores.append(davies_bouldin_score(tfidf_vectors.toarray(), kmeans.labels_))
    return sse, silhouette_scores, davies_bouldin_scores

def plot_cluster_metrics(range_clusters, sse, silhouette_scores, davies_bouldin_scores):
    plt.figure(figsize=(21, 7))
    plt.subplot(1, 3, 1)
    plt.plot(range_clusters, sse, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE (Sum of Squared Errors)")
    plt.title("Elbow Method for Optimal Clusters")

    plt.subplot(1, 3, 2)
    plt.plot(range_clusters, silhouette_scores, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score for Optimal Clusters")

    plt.subplot(1, 3, 3)
    plt.plot(range_clusters, davies_bouldin_scores, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Davies-Bouldin Index")
    plt.title("Davies-Bouldin Index for Optimal Clusters")

    plt.tight_layout()
    plt.show()

def perform_clustering_and_evaluate(tfidf_vectors, optimal_clusters):
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    kmeans.fit(tfidf_vectors)
    silhouette_avg = silhouette_score(tfidf_vectors, kmeans.labels_)
    davies_bouldin_avg = davies_bouldin_score(tfidf_vectors.toarray(), kmeans.labels_)
    return kmeans.labels_, silhouette_avg, davies_bouldin_avg

def main():
    pdf_dir = "pdf"
    abstracts = read_pdfs_from_directory(pdf_dir)
    tfidf_vectors, tfidf_dense = fit_tfidf_vectorizer(abstracts)
    sse, silhouette_scores, davies_bouldin_scores = determine_optimal_clusters(tfidf_vectors, min(25, len(abstracts)))
    plot_cluster_metrics(range(2, min(25, len(abstracts)) + 1), sse, silhouette_scores, davies_bouldin_scores)
    optimal_clusters = 17  # Based on visual inspection of plots
    labels, silhouette_avg, davies_bouldin_avg = perform_clustering_and_evaluate(tfidf_vectors, optimal_clusters)
    print(f"Optimal number of clusters: {optimal_clusters}")
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin_avg}")
    for i, label in enumerate(labels):
        print(f"Abstract {i} is in cluster {label}")

if __name__ == "__main__":
    main()
