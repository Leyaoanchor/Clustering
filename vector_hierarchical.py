import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
import fitz

def read_pdf(file_path):
    document = fitz.open(file_path)  # fitz is another name for PyMuPDF
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
n_clusters = 20
agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')  # Removed affinity
labels = agglomerative.fit_predict(X)

# Evaluate the clustering
silhouette_avg = silhouette_score(tfidf_vectors, labels)
davies_bouldin_avg = davies_bouldin_score(tfidf_vectors.toarray(), labels)

print(f"Number of clusters: {n_clusters}")
print(f"Silhouette Score: {silhouette_avg}")
print(f"Davies-Bouldin Index: {davies_bouldin_avg}")


# Print the resulting clusters
for i, abstract in enumerate(abstracts):
    print(f"Abstract {i} is in cluster {labels[i]}")
