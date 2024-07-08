import os
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE

def read_pdf(file_path):
    document = fitz.open(file_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def read_pdfs_from_directory(directory):
    abstracts = []
    filenames = []  # Store filenames for labeling in the plot
    for f in os.listdir(directory):
        if f.endswith(".pdf"):
            full_path = os.path.join(directory, f)
            abstracts.append(read_pdf(full_path))
            filenames.append(f)  # Capture filename without path
    if not abstracts:
        raise ValueError("No PDF files found in the specified directory.")
    return abstracts, filenames

def fit_tfidf_vectorizer(abstracts):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_vectors = vectorizer.fit_transform(abstracts)
    return tfidf_vectors

def perform_clustering(tfidf_vectors, n_clusters=17):  # Set default based on previous determination
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(tfidf_vectors)
    return labels, kmeans

import plotly.express as px
from sklearn.manifold import TSNE

def visualize_tsne_interactive(tfidf_vectors, labels, filenames):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(tfidf_vectors.toarray())
    
    df_tsne = pd.DataFrame(tsne_results, columns=['t-SNE 1', 't-SNE 2'])
    df_tsne['labels'] = labels
    df_tsne['titles'] = filenames  # Assuming 'filenames' list has titles or relevant identifiers

    fig = px.scatter(df_tsne, x='t-SNE 1', y='t-SNE 2', color='labels', hover_data=['titles'])
    fig.update_traces(marker=dict(size=5, opacity=0.6),
                      selector=dict(mode='markers'))
    fig.show()




def main():
    pdf_dir = "pdf"
    abstracts, filenames = read_pdfs_from_directory(pdf_dir)
    tfidf_vectors = fit_tfidf_vectorizer(abstracts)
    labels, kmeans = perform_clustering(tfidf_vectors)
    visualize_tsne_interactive(tfidf_vectors, labels, filenames)

if __name__ == "__main__":
    main()
