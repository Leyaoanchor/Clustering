#!/usr/bin/env python
# coding: utf-8

# In[1]:


# TD-IDF + K-mean

import os
import fitz
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.manifold import TSNE
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

pdf_dir = "/Users/summerzhao/Documents/cluster/pdf"

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [word for word in tokens if not word in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def read_pdf(file_path):
    document = fitz.open(file_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()

    return preprocess_text(text)


def read_pdfs_from_directory(directory):
    articles = []
    filenames = []  # List to store filenames
    for f in os.listdir(directory):
        if f.endswith(".pdf"):
            articles.append(read_pdf(os.path.join(directory, f)))
            filenames.append(f)
    if len(articles) == 0:
        raise ValueError("No PDF files found in the specified directory.")
    return articles, filenames


def fit_tfidf_vectorizer(articles):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_vectors = vectorizer.fit_transform(articles)
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



def TD_IDF_Kmean():
    pdf_dir = "pdf"
    print("TD-IDF + K-mean")
    articles, filenames = read_pdfs_from_directory(pdf_dir)
    tfidf_vectors, tfidf_dense = fit_tfidf_vectorizer(articles)
    sse, silhouette_scores, davies_bouldin_scores = determine_optimal_clusters(tfidf_vectors, min(25, len(articles)))
    plot_cluster_metrics(range(2, min(25, len(articles)) + 1), sse, silhouette_scores, davies_bouldin_scores)
    optimal_clusters = 21  # Based on visual inspection of plots
    labels, silhouette_avg, davies_bouldin_avg = perform_clustering_and_evaluate(tfidf_vectors, optimal_clusters)
    
    print(f"Optimal number of clusters: {optimal_clusters}")
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin_avg}")
    visualize_tsne_interactive(tfidf_vectors, labels, filenames)
    
    # Create DataFrame with filenames and labels
    results_df = pd.DataFrame({'Filename': filenames, 'Cluster': labels})
    results_df.to_csv('TF-IDF_kmean.csv', index=False)
    print("Results saved to 'TF-IDF_kmean.csv'.")

    # Optional: Print each article's cluster assignment
    #for i, label in enumerate(labels):
        #print(f"Article {i} is in cluster {label}")



TD_IDF_Kmean()




# In[2]:


#TF-IDF_hierarchical

def compute_tfidf_vectors(abstracts):
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    return tfidf_vectorizer.fit_transform(abstracts)

def compute_distance_matrix(tfidf_vectors):
    similarity_matrix = cosine_similarity(tfidf_vectors)
    return 1 - similarity_matrix  # Convert similarity to distance

def plot_dendrogram(linkage_matrix):
    plt.figure(figsize=(15, 10))
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()


def evaluate_clustering(labels, tfidf_vectors):
    silhouette_avg = silhouette_score(tfidf_vectors, labels)
    davies_bouldin_avg = davies_bouldin_score(tfidf_vectors.toarray(), labels)
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin_avg}")
    return silhouette_avg, davies_bouldin_avg


    
def TF_IDF_hierarchical():
    articles, filenames = read_pdfs_from_directory(pdf_dir)
    tfidf_vectors = compute_tfidf_vectors(articles)

    # Hierarchical clustering setup
    distance_matrix = compute_distance_matrix(tfidf_vectors)
    linkage_matrix = linkage(distance_matrix, method='average')
    plot_dendrogram(linkage_matrix)

    # Decide on the number of clusters from dendrogram or other method
    n_clusters = 9
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    labels = clustering_model.fit_predict(tfidf_vectors.toarray())

    # Evaluate and visualize
    evaluate_clustering(labels, tfidf_vectors)
    visualize_tsne_interactive(tfidf_vectors, labels, filenames)
    

TF_IDF_hierarchical()



# In[5]:


# Longformer + K-mean

from transformers import LongformerModel, LongformerTokenizer

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def get_longformer_embeddings(articles):
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
    embeddings = []
    for text in articles:
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=4096)
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].detach().numpy().flatten())  # Use the CLS token's embeddings
    return np.array(embeddings)

def visualize_tsne_interactive(embeddings, labels, filenames):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    
    df_tsne = pd.DataFrame(tsne_results, columns=['t-SNE 1', 't-SNE 2'])
    df_tsne['labels'] = labels
    df_tsne['titles'] = filenames

    fig = px.scatter(df_tsne, x='t-SNE 1', y='t-SNE 2', color='labels', hover_data=['titles'])
    fig.update_traces(marker=dict(size=5, opacity=0.6),
                      selector=dict(mode='markers'))
    fig.show()


def Longformer_Kmean():
    pdf_dir = "pdf"
    print("Longformer + K-mean")
    articles, filenames = read_pdfs_from_directory(pdf_dir)
    embeddings = get_longformer_embeddings(articles)
    sse, silhouette_scores, davies_bouldin_scores = determine_optimal_clusters(embeddings, min(25, len(articles)))
    plot_cluster_metrics(range(2, min(25, len(articles)) + 1), sse, silhouette_scores, davies_bouldin_scores)
    optimal_clusters = 9  # Based on visual inspection of plots
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(embeddings, kmeans.labels_)
    davies_bouldin_avg = davies_bouldin_score(embeddings, kmeans.labels_)
    print(f"Optimal number of clusters: {optimal_clusters}")
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin_avg}")

    visualize_tsne_interactive(embeddings, labels, filenames)
    
    results_df = pd.DataFrame({'Filename': filenames, 'Cluster': labels})
    results_df.to_csv('Longformer_kmean.csv', index=False)
    print("Results saved to 'Longformer_kmean.csv'.")

Longformer_Kmean()




# In[6]:


# BERT + K-mean

def preprocess_text(text):
    # Simpler preprocessing since BERT handles context better
    text = re.sub(r'\d+', '', text)  # remove digits
    return text.strip()

def compute_bert_embeddings(articles):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # 
    embeddings = model.encode(articles, convert_to_tensor=False) 
    return embeddings

def determine_optimal_clusters(embeddings, max_clusters):
    sse = []
    silhouette_scores = []
    davies_bouldin_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(embeddings, kmeans.labels_))
        davies_bouldin_scores.append(davies_bouldin_score(embeddings, kmeans.labels_))
    return sse, silhouette_scores, davies_bouldin_scores

def visualize_tsne_interactive(embeddings, labels, filenames):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    
    df_tsne = pd.DataFrame(tsne_results, columns=['t-SNE 1', 't-SNE 2'])
    df_tsne['labels'] = labels
    df_tsne['titles'] = filenames

    fig = px.scatter(df_tsne, x='t-SNE 1', y='t-SNE 2', color='labels', hover_data=['titles'])
    fig.update_traces(marker=dict(size=5, opacity=0.6),
                      selector=dict(mode='markers'))
    fig.show()
    
def Bert_Kmean():
    pdf_dir = "pdf"
    print("BERT + K-mean")
    articles, filenames = read_pdfs_from_directory(pdf_dir)
    embeddings = compute_bert_embeddings(articles)
    sse, silhouette_scores, davies_bouldin_scores = determine_optimal_clusters(embeddings, min(25, len(articles)))
    plot_cluster_metrics(range(2, min(25, len(articles)) + 1), sse, silhouette_scores, davies_bouldin_scores)
    optimal_clusters = 9  # Based on visual inspection of plots
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    kmeans.fit(embeddings)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(embeddings, labels)
    davies_bouldin_avg = davies_bouldin_score(embeddings, labels)
    
    print(f"Optimal number of clusters: {optimal_clusters}")
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin_avg}")
    visualize_tsne_interactive(embeddings, labels, filenames)
    
    results_df = pd.DataFrame({'Filename': filenames, 'Cluster': labels})
    results_df.to_csv('BERT_kmean.csv', index=False)
    print("Results saved to 'BERT_kmean.csv'.")

Bert_Kmean()


# In[7]:


#Bert + hierarchical


def Bert_Hierarchical():
    articles, filenames = read_pdfs_from_directory(pdf_dir)
    embeddings = compute_bert_embeddings(articles)

    # Hierarchical clustering setup
    # Using Euclidean distance as linkage requires true metric
    linkage_matrix = linkage(embeddings, method='ward')
    plot_dendrogram(linkage_matrix)

    # Decide on the number of clusters from dendrogram or other method
    n_clusters = 9
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clustering_model.fit_predict(embeddings)

    # Evaluate and visualize
    silhouette_avg = silhouette_score(embeddings, labels)
    davies_bouldin_avg = davies_bouldin_score(embeddings, labels)
    print(f"Number of clusters: {n_clusters}")
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin_avg}")
    
    visualize_tsne_interactive(embeddings, labels, filenames)
    
    results_df = pd.DataFrame({'Filename': filenames, 'Cluster': labels})
    results_df.to_csv('Bert_Hierarchical.csv', index=False)
    print("Results saved to 'Bert_Hierarchical.csv'.")


Bert_Hierarchical()


# In[8]:


#Longformer_Hierarchical

from transformers import LongformerModel, LongformerTokenizer
import torch

def compute_longformer_embeddings(articles):
    tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    model = LongformerModel.from_pretrained('allenai/longformer-base-4096')

    embeddings = []
    for article in articles:
        encoded_input = tokenizer(article, return_tensors='pt', max_length=4096, truncation=True, padding='max_length')
        output = model(**encoded_input)
        # Take the mean of the last hidden state to get a single vector per article
        article_embedding = output.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.append(article_embedding.squeeze())  # Remove unnecessary dimensions

    return np.array(embeddings)


def Longformer_Hierarchical():
    articles, filenames = read_pdfs_from_directory(pdf_dir)
    embeddings = compute_longformer_embeddings(articles)

    # Hierarchical clustering setup
    linkage_matrix = linkage(embeddings, method='ward')
    plot_dendrogram(linkage_matrix)

    # Decide on the number of clusters from dendrogram or other method
    n_clusters = 9
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clustering_model.fit_predict(embeddings)

    # Evaluate and visualize
    silhouette_avg = silhouette_score(embeddings, labels)
    davies_bouldin_avg = davies_bouldin_score(embeddings, labels)
    print(f"Number of clusters: {n_clusters}")
    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin_avg}")
    
    visualize_tsne_interactive(embeddings, labels, filenames)
        
    results_df = pd.DataFrame({'Filename': filenames, 'Cluster': labels})
    results_df.to_csv('Longformer_Hierarchical.csv', index=False)
    print("Results saved to 'Longformer_Hierarchical.csv'.")

Longformer_Hierarchical()


# In[ ]:





# In[9]:


#BERT + DBSACAN

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN



def plot_k_distance_graph(embeddings, k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    distance_dec = sorted(distances[:, k-1], reverse=True)
    plt.figure(figsize=(8, 4))
    plt.plot(distance_dec)
    plt.xlabel('Points')
    plt.ylabel('Distance')
    plt.title('K-Distance Graph')
    plt.show()
    



def apply_dbscan(embeddings, eps, min_samples):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
    labels = clustering.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    if n_clusters_ > 2:
        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("eps: ", eps," min sameples: ", min_samples)
        print()
    return labels



def Bert_DBSACAN():
    pdf_dir = "pdf"
    print("BERT + DBSACAN")
    articles, filenames = read_pdfs_from_directory(pdf_dir)
    embeddings = compute_bert_embeddings(articles)
    plot_k_distance_graph(embeddings, 5)
    for i in range(20,200):
        for j in range(2,8):
            labels = apply_dbscan(embeddings, eps=i*1.0/100, min_samples=j)
            
            if i*1.0/100 == 0.91 and j == 2:        
                visualize_tsne_interactive(embeddings, labels, filenames)
                results_df = pd.DataFrame({'Filename': filenames, 'Cluster': labels})
                results_df.to_csv('Results_dbscan.csv', index=False)
                print("Results saved to 'Results_dbscan.csv'.")
    

Bert_DBSACAN()


# In[ ]:





# In[10]:


#Automatically extract article data and return Python code

import google.generativeai as genai


GOOGLE_API_KEY = "API"
genai.configure(api_key=GOOGLE_API_KEY)

path = "/Users/summerzhao/Documents/cluster/txt/A Perspective on the Overarching Role of Hydrogen, Ammonia, and Methanol Carbon-Neutral Fuels towards Net Zero Emission in the Next Three Decades.pdf.txt"

txt = open(f"{path}","r").read()

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content(
    f"give me the simple python code to visualize the data from the following article, no more than two plot, only give me the python code no need others:{txt}",
    generation_config=genai.types.GenerationConfig(
        # Only one candidate for now.
        candidate_count=1,
        stop_sequences= ["```\n"],
        max_output_tokens=2000,
        temperature=1.0,
    ),
)

cleaned_txt = re.sub(r'^\s*```python\s*\n', '', response.text, flags=re.MULTILINE)
cleaned_txt = cleaned_txt[:-3]

exec(cleaned_txt)






