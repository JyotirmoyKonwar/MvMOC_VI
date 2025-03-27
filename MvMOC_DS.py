import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import math
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt_tab')

# --------------------------
# Text Processing Functions
# --------------------------

def read_and_tokenize_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    sentences = sent_tokenize(text)
    return sentences

# --------------------------
# Embedding Functions
# --------------------------

def get_tfidf_embeddings(sentences):
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return tfidf_matrix.toarray()

def get_bert_embeddings(sentences):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    embeddings = model.encode(sentences)
    return embeddings

def get_colbert_embeddings(sentences):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    
    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use CLS token representation as sentence embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls_embedding.squeeze())
    
    return np.array(embeddings)

# --------------------------
# Clustering Metrics
# --------------------------

def pbm_index(X, centers, labels):
    """
    PBM (Point-Biserial Metric) index for clustering validation
    Higher values indicate better clustering
    """
    k = len(centers)
    N = X.shape[0]
    
    # Calculate E1 - total dispersion when all data in one cluster
    global_center = np.mean(X, axis=0)
    E1 = np.sum(np.linalg.norm(X - global_center, axis=1))
    
    # Calculate Ew - within-cluster dispersion
    Ew = 0.0
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            Ew += np.sum(np.linalg.norm(cluster_points - centers[i], axis=1))
    
    # Calculate Db - between-cluster dispersion
    Db = 0.0
    for i in range(k):
        for j in range(i+1, k):
            Db += np.linalg.norm(centers[i] - centers[j])
    
    if Ew == 0:
        return 0
    
    pbm = (E1 * Db / (k * Ew)) ** 2
    return pbm

def calculate_silhouette(X, labels):
    if len(np.unique(labels)) == 1:
        return 0  # Minimum value for silhouette score
    try:
        return silhouette_score(X, labels)
    except:
        return 0

# --------------------------
# Multi-view Clustering
# --------------------------

def multi_view_clustering(views, n_clusters):
    all_labels = []
    all_centers = []
    
    for view in views:
        # Standardize each view
        scaler = StandardScaler()
        X = scaler.fit_transform(view)
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_
        
        all_labels.append(labels)
        all_centers.append(centers)
    
    return all_labels, all_centers

# --------------------------
# Optimization Problem
# --------------------------

class MultiViewClusteringProblem(Problem):
    def __init__(self, views, min_clusters=2, max_clusters=10):
        self.views = views
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        super().__init__(n_var=1, n_obj=2, n_constr=0, 
                         xl=np.array([min_clusters]), xu=np.array([max_clusters]))

    def _evaluate(self, x, out, *args, **kwargs):
        silhouettes = []
        pbms = []
        
        for i in range(x.shape[0]):
            n_clusters = int(x[i, 0])
            
            # Perform multi-view clustering
            labels_list, centers_list = multi_view_clustering(self.views, n_clusters)
            
            # Calculate objectives for each view and average them
            view_silhouettes = []
            view_pbms = []
            
            for view_idx, view in enumerate(self.views):
                labels = labels_list[view_idx]
                centers = centers_list[view_idx]
                
                # Standardize the view
                scaler = StandardScaler()
                X = scaler.fit_transform(view)
                
                # Calculate metrics
                sil = calculate_silhouette(X, labels)
                pbm = pbm_index(X, centers, labels)
                
                view_silhouettes.append(sil)
                view_pbms.append(pbm)
            
            # Average the metrics across views
            avg_silhouette = np.mean(view_silhouettes)
            avg_pbm = np.mean(view_pbms)
            
            silhouettes.append(avg_silhouette)
            pbms.append(avg_pbm)
        
        # We want to maximize both objectives
        out["F"] = np.column_stack([-np.array(silhouettes), -np.array(pbms)])

# --------------------------
# Main Execution
# --------------------------

def main(text_file_path):
    # 1. Read and preprocess text
    sentences = read_and_tokenize_text(text_file_path)
    print(f"Processed {len(sentences)} sentences.")
    
    # 2. Generate multi-view embeddings
    print("Generating TF-IDF embeddings...")
    tfidf_embeddings = get_tfidf_embeddings(sentences)
    
    print("Generating BERT embeddings...")
    bert_embeddings = get_bert_embeddings(sentences)
    
    print("Generating ColBERT embeddings...")
    colbert_embeddings = get_colbert_embeddings(sentences)
    
    # 3. Prepare views
    views = [tfidf_embeddings, bert_embeddings, colbert_embeddings]
    
    # 4. Set up optimization problem
    problem = MultiViewClusteringProblem(views, min_clusters=2, max_clusters=10)
    
    # 5. Configure algorithm
    algorithm = NSGA2(
        pop_size=20,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    
    termination = get_termination("n_gen", 30)
    
    # 6. Run optimization
    print("Starting optimization...")
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=42,
                   save_history=True,
                   verbose=True)
    
    # 7. Process results
    print("Optimization complete.")
    print("Optimal solutions:")
    for i in range(len(res.F)):
        n_clusters = int(res.X[i, 0])
        silhouette = -res.F[i, 0]
        pbm = -res.F[i, 1]
        print(f"Solution {i+1}: Clusters={n_clusters}, Silhouette={silhouette:.4f}, PBM={pbm:.4f}")
    
    # 8. Select the best compromise solution (you can modify this selection criteria)
    best_idx = np.argmax([-res.F[i, 0] - res.F[i, 1] for i in range(len(res.F))])
    best_n_clusters = int(res.X[best_idx, 0])
    print(f"\nSelected best solution: {best_n_clusters} clusters")
    
    # 9. Perform final clustering with optimal number of clusters
    final_labels, final_centers = multi_view_clustering(views, best_n_clusters)
    
    # 10. Return results
    results = {
        'sentences': sentences,
        'optimal_clusters': best_n_clusters,
        'tfidf_labels': final_labels[0],
        'bert_labels': final_labels[1],
        'colbert_labels': final_labels[2],
        'all_solutions': [(int(x[0]), -f[0], -f[1]) for x, f in zip(res.X, res.F)]
    }
    
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python multiview_clustering.py <text_file_path>")
        sys.exit(1)
    
    results = main(sys.argv[1])
    
    # Print some sample results
    print("\nSample clustering results:")
    for i in range(min(10, len(results['sentences']))):
        print(f"Sentence {i+1}:")
        print(results['sentences'][i])
        print(f"TF-IDF cluster: {results['tfidf_labels'][i]}")
        print(f"BERT cluster: {results['bert_labels'][i]}")
        print(f"ColBERT cluster: {results['colbert_labels'][i]}")
        print()