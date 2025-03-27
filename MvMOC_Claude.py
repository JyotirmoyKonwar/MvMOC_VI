import numpy as np
import pandas as pd
import torch
import nltk
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer

# Pymoo imports
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.evaluation import Evaluation
from pymoo.core.termination import DefaultSingleObjectiveTermination

# Download NLTK resources
nltk.download('punkt', quiet=True)

class MultiViewMultiObjectiveClustering:
    def __init__(self, text):
        # Sentence tokenization
        self.sentences = nltk.sent_tokenize(text)
        
        # Initialize embedding models
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.colbert_model = SentenceTransformer('colbert-ir/colbert-412M-msmarco')
        
        # Prepare embeddings
        self.prepare_embeddings()
    
    def prepare_embeddings(self):
        # TF-IDF Embedding
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.sentences).toarray()
        
        # BERT Embedding
        bert_embeddings = []
        for sentence in self.sentences:
            inputs = self.bert_tokenizer(sentence, return_tensors='pt', 
                                         truncation=True, 
                                         max_length=512, 
                                         padding=True)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).numpy()
                bert_embeddings.append(embedding[0])
        self.bert_matrix = np.array(bert_embeddings)
        
        # ColBERT Embedding
        self.colbert_matrix = self.colbert_model.encode(self.sentences)
        
        # Standardize embeddings
        self.scaler_tfidf = StandardScaler()
        self.scaler_bert = StandardScaler()
        self.scaler_colbert = StandardScaler()
        
        self.tfidf_scaled = self.scaler_tfidf.fit_transform(self.tfidf_matrix)
        self.bert_scaled = self.scaler_bert.fit_transform(self.bert_matrix)
        self.colbert_scaled = self.scaler_colbert.fit_transform(self.colbert_matrix)
    
    def calculate_pbm_index(self, labels, embeddings):
        """
        Calculate the PBM (Point Biserial) Index
        """
        # Average of cluster centroids
        unique_labels = np.unique(labels)
        centroids = np.array([embeddings[labels == i].mean(axis=0) for i in unique_labels])
        global_centroid = embeddings.mean(axis=0)
        
        # Between-cluster distances
        between_cluster_distances = np.linalg.norm(centroids - global_centroid, axis=1)
        
        # Within-cluster distances
        within_cluster_distances = []
        for i in unique_labels:
            cluster_points = embeddings[labels == i]
            cluster_centroid = cluster_points.mean(axis=0)
            distances = np.linalg.norm(cluster_points - cluster_centroid, axis=1)
            within_cluster_distances.append(np.mean(distances))
        
        # PBM Index calculation
        pbm_index = (np.mean(between_cluster_distances) / np.mean(within_cluster_distances)) ** 2
        return pbm_index
    
    def perform_clustering(self, n_clusters_list):
        """
        Perform clustering for multiple views
        """
        # Unpack cluster numbers for each view
        n_clusters_tfidf, n_clusters_bert, n_clusters_colbert = n_clusters_list
        
        # Clustering for each view
        kmeans_tfidf = KMeans(n_clusters=max(2, n_clusters_tfidf), n_init=10, random_state=42)
        labels_tfidf = kmeans_tfidf.fit_predict(self.tfidf_scaled)
        
        kmeans_bert = KMeans(n_clusters=max(2, n_clusters_bert), n_init=10, random_state=42)
        labels_bert = kmeans_bert.fit_predict(self.bert_scaled)
        
        kmeans_colbert = KMeans(n_clusters=max(2, n_clusters_colbert), n_init=10, random_state=42)
        labels_colbert = kmeans_colbert.fit_predict(self.colbert_scaled)
        
        # Calculate objectives
        objectives = []
        
        # Silhouette Scores
        sil_tfidf = silhouette_score(self.tfidf_scaled, labels_tfidf)
        sil_bert = silhouette_score(self.bert_scaled, labels_bert)
        sil_colbert = silhouette_score(self.colbert_scaled, labels_colbert)
        
        # PBM Indices
        pbm_tfidf = self.calculate_pbm_index(labels_tfidf, self.tfidf_scaled)
        pbm_bert = self.calculate_pbm_index(labels_bert, self.bert_scaled)
        pbm_colbert = self.calculate_pbm_index(labels_colbert, self.colbert_scaled)
        
        return [
            -sil_tfidf,     # Negative because we want to maximize
            -sil_bert, 
            -sil_colbert, 
            -pbm_tfidf, 
            -pbm_bert, 
            -pbm_colbert
        ]

class MultiViewClusteringProblem(Problem):
    def __init__(self, clustering_instance):
        """
        Define the multi-objective optimization problem
        """
        super().__init__(
            n_var=3,           # Number of variables (clusters for each view)
            n_obj=6,           # Number of objectives
            n_constr=0,        # Number of constraints
            xl=np.array([2, 2, 2]),     # Lower bounds for cluster numbers
            xu=np.array([10, 10, 10])   # Upper bounds for cluster numbers
        )
        self.clustering_instance = clustering_instance
    
    def _evaluate(self, X, *args, **kwargs):
        """
        Evaluate the objective functions for each solution
        """
        # Store results for parallel evaluation
        F = np.full((X.shape[0], self.n_obj), np.nan)
        
        # Evaluate each solution
        for i, solution in enumerate(X):
            F[i, :] = self.clustering_instance.perform_clustering(solution)
        
        return F

def read_text_from_file(file_path):
    """
    Read text from a file
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def main():
    # Read text from file
    file_path = 'text.txt'  # Replace with your text file path
    text = read_text_from_file(file_path)
    
    # Initialize multi-view clustering
    mv_clustering = MultiViewMultiObjectiveClustering(text)
    
    # Create multi-objective problem
    problem = MultiViewClusteringProblem(mv_clustering)
    
    # Define NSGA-II algorithm
    algorithm = NSGA2(
        pop_size=50,
        sampling='real',
        mutation=('real', {'prob': 0.2}),
        crossover=('real', {'prob': 0.8})
    )
    
    # Termination criteria
    termination = DefaultSingleObjectiveTermination(
        n_max_iterations=100
    )
    
    # Run optimization
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=True
    )
    
    # Analyze results
    print("\nOptimal Solutions:")
    for i, solution in enumerate(res.X):
        print(f"Solution {i+1}:")
        print(f"Clusters (TF-IDF, BERT, ColBERT): {solution}")
        
        # Validate clustering for this solution
        objectives = mv_clustering.perform_clustering(solution)
        
        print("Objectives:")
        print(f"  Silhouette Scores: {-objectives[0]:.4f} (TF-IDF), {-objectives[1]:.4f} (BERT), {-objectives[2]:.4f} (ColBERT)")
        print(f"  PBM Indices: {-objectives[3]:.4f} (TF-IDF), {-objectives[4]:.4f} (BERT), {-objectives[5]:.4f} (ColBERT)")
        print()
    
    # Best solution analysis
    best_solution_idx = np.argmin(res.F.mean(axis=1))
    best_solution = res.X[best_solution_idx]
    print("\nBest Overall Solution:")
    print(f"Optimal Clusters (TF-IDF, BERT, ColBERT): {best_solution}")

if __name__ == "__main__":
    main()