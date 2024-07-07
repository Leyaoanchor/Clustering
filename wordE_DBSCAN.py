import os
import fitz
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score


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

# Perform DBSCAN clustering with parameter tuning
eps_values = [0.3, 0.5, 0.7, 0.9]  # Different eps values to try
min_samples_values = [2, 3, 5]  # Different min_samples values to try

best_silhouette = -1
best_davies_bouldin = float("inf")
best_labels = None
best_eps = None
best_min_samples = None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(embeddings)

        # Filter out noise points (label = -1) for evaluation
        filtered_labels = labels[labels != -1]
        filtered_embeddings = embeddings[labels != -1]

        if (
            len(set(filtered_labels)) > 1
        ):  # Ensure there are enough clusters for evaluation
            silhouette_avg = silhouette_score(filtered_embeddings, filtered_labels)
            davies_bouldin_avg = davies_bouldin_score(
                filtered_embeddings, filtered_labels
            )
            if (
                silhouette_avg > best_silhouette
                and davies_bouldin_avg < best_davies_bouldin
            ):
                best_silhouette = silhouette_avg
                best_davies_bouldin = davies_bouldin_avg
                best_labels = labels
                best_eps = eps
                best_min_samples = min_samples

# Evaluate the best clustering result
if best_labels is not None:
    print(f"Best DBSCAN parameters: eps={best_eps}, min_samples={best_min_samples}")
    print(
        f"Number of clusters: {len(set(best_labels)) - (1 if -1 in best_labels else 0)}"
    )
    print(f"Best Silhouette Score: {best_silhouette}")
    print(f"Best Davies-Bouldin Index: {best_davies_bouldin}")

    # Print the resulting clusters
    for i, abstract in enumerate(abstracts):
        print(f"Abstract {i} is in cluster {best_labels[i]}")
else:
    print(
        "DBSCAN did not find any suitable clusters. Try different parameters or preprocess the data."
    )
