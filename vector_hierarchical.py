import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF

def read_pdf(file_path):
    document = fitz.open(file_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Load PDF files and extract text
abstracts = []
pdf_dir = "pdf"
for f in os.listdir(pdf_dir):
    if f.endswith(".pdf"):
        abstracts.append(read_pdf(os.path.join(pdf_dir, f)))

if len(abstracts) == 0:
    raise ValueError("No PDF files found in the specified directory.")

# Create a TF-IDF vectorizer and transform the abstracts
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_vectors = tfidf_vectorizer.fit_transform(abstracts)

# Convert cosine similarity to cosine distance
X = tfidf_vectors.toarray()
similarity_matrix = cosine_similarity(X)
distance_matrix = 1 - similarity_matrix  # Converting similarity to distance

# Generate the linkage matrix
linkage_matrix = linkage(distance_matrix, method='average')  # Changed method

# Plot the Dendrogram
plt.figure(figsize=(15, 10))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# Perform Agglomerative Hierarchical Clustering
n_clusters = 9
agglomerative_initial = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
labels_initial = agglomerative_initial.fit_predict(X)
silhouette_avg_initial = silhouette_score(tfidf_vectors, labels_initial)
davies_bouldin_avg_initial = davies_bouldin_score(tfidf_vectors.toarray(), labels_initial)

print(f"Initial number of clusters: {n_clusters}")
print(f"Initial Silhouette Score: {silhouette_avg_initial}")
print(f"Initial Davies-Bouldin Index: {davies_bouldin_avg_initial}")


range_clusters = range(2, 21)  # Testing from 2 to 20 clusters
silhouette_scores = []
davies_bouldin_scores = []

for n_clusters in range_clusters:
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    labels = agglomerative.fit_predict(X)
    silhouette_avg = silhouette_score(tfidf_vectors, labels)
    davies_bouldin_avg = davies_bouldin_score(tfidf_vectors.toarray(), labels)
    silhouette_scores.append(silhouette_avg)
    davies_bouldin_scores.append(davies_bouldin_avg)


# Plot Silhouette Score
plt.figure(figsize=(10, 5))
plt.plot(range_clusters, silhouette_scores, marker='o', label='Silhouette Score')
plt.title('Silhouette Score by Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(range_clusters)
plt.legend()
plt.grid(True)
plt.show()

# Plot Davies-Bouldin Index
plt.figure(figsize=(10, 5))
plt.plot(range_clusters, davies_bouldin_scores, marker='o', label='Davies-Bouldin Index')
plt.title('Davies-Bouldin Index by Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Davies-Bouldin Index')
plt.xticks(range_clusters)
plt.legend()
plt.grid(True)
plt.show()


agglomerative_final = AgglomerativeClustering(n_clusters=9, linkage='average')
final_labels = agglomerative_final.fit_predict(X)
for i, abstract in enumerate(abstracts):
    print(f"Abstract {i} is in final cluster {final_labels[i]}")
