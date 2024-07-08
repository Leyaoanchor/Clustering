import os
import csv
import fitz  # PyMuPDF
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

def read_pdf(file_path):
    document = fitz.open(file_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to write cluster assignments to a CSV file
def write_clusters_to_csv(filenames, labels, file_path='wordE_kmean.csv'):
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'Cluster'])
        for filename, label in zip(filenames, labels):
            writer.writerow([filename, label])

# Load PDF files and extract text
abstracts = []
filenames = []  # List to store filenames
pdf_dir = "pdf"  # Directory containing the PDF files
for f in os.listdir(pdf_dir):
    if f.endswith(".pdf"):  # Ensure we only process PDF files
        full_path = os.path.join(pdf_dir, f)
        abstracts.append(read_pdf(full_path))
        filenames.append(f)  # Store the filename

# Check if abstracts were extracted
if len(abstracts) == 0:
    raise ValueError("No PDF files found in the specified directory.")

# Load pre-trained BERT model for sentence embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert abstracts to BERT embeddings
embeddings = model.encode(abstracts, show_progress_bar=True)

# Determine the optimal number of clusters using the elbow method and silhouette score
sse = []
silhouette_scores = []
davies_bouldin_scores = []

max_clusters = min(25, len(abstracts))  # Ensure max_clusters does not exceed the number of samples
for k in range(2, max_clusters + 1):  # Silhouette score requires at least 2 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings)
    sse.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(embeddings, kmeans.labels_))
    davies_bouldin_scores.append(davies_bouldin_score(embeddings, kmeans.labels_))

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

# Choose the optimal number of clusters (e.g., from the plots)
optimal_clusters = 11  # Adjust this value based on the plots
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(embeddings)

# Output the filename and cluster assignments to CSV
write_clusters_to_csv(filenames, kmeans.labels_)

print(f"Optimal number of clusters: {optimal_clusters}")
print(f"Silhouette Score: {silhouette_score(embeddings, kmeans.labels_)}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(embeddings, kmeans.labels_)}")
print(f"Cluster assignments have been saved to 'wordE_kmean.csv'")
