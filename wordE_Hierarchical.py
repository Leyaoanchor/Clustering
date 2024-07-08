import os
import fitz
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import cosine_similarity

def read_pdf(file_path):
    document = fitz.open(file_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Load PDF files and extract text
abstracts = []
pdf_dir = "pdf"  # Directory containing the PDF files
for f in os.listdir(pdf_dir):
    if f.endswith(".pdf"):  # Ensure we only process PDF files
        abstracts.append(read_pdf(os.path.join(pdf_dir, f)))

# Check if abstracts were extracted
if len(abstracts) == 0:
    raise ValueError("No PDF files found in the specified directory.")

# Load pre-trained BERT model for sentence embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert abstracts to BERT embeddings
embeddings = model.encode(abstracts, show_progress_bar=True)

# Convert cosine similarity to cosine distance
similarity_matrix = cosine_similarity(embeddings)
distance_matrix = 1 - similarity_matrix  # Converting similarity to distance

# Generate the linkage matrix
linkage_matrix = linkage(distance_matrix, method='average')

# Plot the Dendrogram
plt.figure(figsize=(15, 10))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

# Plot the elbow curve, silhouette scores, and Davies-Bouldin scores
plt.figure(figsize=(21, 7))
plt.subplot(1, 3, 1)
plt.plot(range(2, max_clusters + 1), sse, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("SSE (Sum of Squared Errors)")
plt.title("Elbow Method for Optimal Clusters")

plt.subplot(1, 3, 2)
plt.plot(range(2, max_clusters + 1), silhouette_scores, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Optimal Clusters")

plt.subplot(1, 3, 3)
plt.plot(range(2, max_clusters + 1), davies_bouldin_scores, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Davies-Bouldin Index")
plt.title("Davies-Bouldin Index for Optimal Clusters")

plt.tight_layout()
plt.show()

# Perform Agglomerative Hierarchical Clustering
n_clusters = 11  # Adjust this number based on your data
agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
labels = agglomerative.fit_predict(embeddings)

# Evaluate the quality of the clustering using Davies-Bouldin index
davies_bouldin_avg = davies_bouldin_score(embeddings, labels)

print(f"Number of clusters: {n_clusters}")
print(f"Davies-Bouldin Index: {davies_bouldin_avg}")

# Print the resulting clusters
for i, abstract in enumerate(abstracts):
    print(f"Abstract {i} is in cluster {labels[i]}")
