import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import math
from sklearn.preprocessing import StandardScaler
import re

# --------------------------
# Text Processing Functions
# --------------------------

def read_and_tokenize_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Enhanced sentence splitting that handles multiple punctuation marks and newlines
    sentences = []
    # First split by newlines
    lines = text.split('\n')
    for line in lines:
        # Then split by sentence-ending punctuation
        chunks = re.split(r'(?<=[.!?])\s+', line.strip())
        for chunk in chunks:
            # Further split if multiple sentences were joined together
            sub_sentences = re.split(r'(?<=[.!?])(?=[^\s])', chunk)
            sentences.extend([s.strip() for s in sub_sentences if s.strip()])
    
    # Filter out empty strings and very short sentences
    sentences = [s for s in sentences if len(s) > 3]  # Minimum 3 characters
    
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
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls_embedding.squeeze())
    
    return np.array(embeddings)

# --------------------------
# Clustering Metrics
# --------------------------

def pbm_index(X, centers, labels):
    k = len(centers)
    N = X.shape[0]
    
    global_center = np.mean(X, axis=0)
    E1 = np.sum(np.linalg.norm(X - global_center, axis=1))
    
    Ew = 0.0
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            Ew += np.sum(np.linalg.norm(cluster_points - centers[i], axis=1))
    
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
        return 0
    try:
        return silhouette_score(X, labels)
    except:
        return 0

# --------------------------
# Multi-view Clustering
# --------------------------

def multi_view_clustering(concatenated_embeddings, n_clusters):
    scaler = StandardScaler()
    X = scaler.fit_transform(concatenated_embeddings)
    
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42, metric='euclidean')
    labels = kmedoids.fit_predict(X)
    centers = kmedoids.cluster_centers_
    
    return labels, centers

# --------------------------
# Optimization Problem
# --------------------------

class MultiViewClusteringProblem(Problem):
    def __init__(self, concatenated_embeddings, min_clusters=2, max_clusters=10):
        self.concatenated_embeddings = concatenated_embeddings
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        super().__init__(n_var=1, n_obj=2, n_constr=0, 
                       xl=np.array([min_clusters]), xu=np.array([max_clusters]))

    def _evaluate(self, x, out, *args, **kwargs):
        silhouettes = []
        pbms = []
        
        for i in range(x.shape[0]):
            n_clusters = int(x[i, 0])
            
            labels, centers = multi_view_clustering(self.concatenated_embeddings, n_clusters)
            
            sil = calculate_silhouette(self.concatenated_embeddings, labels)
            pbm = pbm_index(self.concatenated_embeddings, centers, labels)
            
            silhouettes.append(sil)
            pbms.append(pbm)
        
        out["F"] = np.column_stack([-np.array(silhouettes), -np.array(pbms)])

# --------------------------
# Main Execution
# --------------------------

def main(text_file_path):
    # 1. Read and preprocess text with enhanced splitting
    sentences = read_and_tokenize_text(text_file_path)
    print(f"Processed {len(sentences)} sentences.")
    print("Sample sentences:")
    for s in sentences[:5]:
        print(f"- {s}")
    
    # 2. Generate multi-view embeddings
    print("\nGenerating TF-IDF embeddings...")
    tfidf_embeddings = get_tfidf_embeddings(sentences)
    
    print("Generating BERT embeddings...")
    bert_embeddings = get_bert_embeddings(sentences)
    
    print("Generating ColBERT embeddings...")
    colbert_embeddings = get_colbert_embeddings(sentences)
    
    # 3. Concatenate all embeddings
    print("\nConcatenating embeddings...")
    concatenated_embeddings = np.concatenate([
        tfidf_embeddings,
        bert_embeddings,
        colbert_embeddings
    ], axis=1)
    print(f"Final embedding dimension: {concatenated_embeddings.shape[1]}")
    
    # 4. Set up optimization problem
    problem = MultiViewClusteringProblem(concatenated_embeddings, min_clusters=2, max_clusters=10)
    
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
    print("\nStarting optimization...")
    res = minimize(problem,
                  algorithm,
                  termination,
                  seed=42,
                  save_history=True,
                  verbose=True)
    
    # 7. Process results
    print("\nOptimization complete.")
    print("Optimal solutions:")
    for i in range(len(res.F)):
        n_clusters = int(res.X[i, 0])
        silhouette = -res.F[i, 0]
        pbm = -res.F[i, 1]
        print(f"Solution {i+1}: Clusters={n_clusters}, Silhouette={silhouette:.4f}, PBM={pbm:.4f}")
    
    # 8. Select the best compromise solution
    best_idx = np.argmax([-res.F[i, 0] - res.F[i, 1] for i in range(len(res.F))])
    best_n_clusters = int(res.X[best_idx, 0])
    print(f"\nSelected best solution: {best_n_clusters} clusters")
    
    # 9. Perform final clustering
    final_labels, final_centers = multi_view_clustering(concatenated_embeddings, best_n_clusters)
    
    # 10. Return results
    results = {
        'sentences': sentences,
        'optimal_clusters': best_n_clusters,
        'labels': final_labels,
        'all_solutions': [(int(x[0]), -f[0], -f[1]) for x, f in zip(res.X, res.F)],
        'embeddings_shape': concatenated_embeddings.shape
    }
    
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python multiview_clustering.py <text_file_path>")
        sys.exit(1)
    
    results = main(sys.argv[1])
    
    # Print clustering results
    print("\nClustering results summary:")
    print(f"Optimal number of clusters: {results['optimal_clusters']}")
    print(f"Total sentences clustered: {len(results['sentences'])}")
    print(f"Embedding dimension: {results['embeddings_shape'][1]}")
    
    # Print cluster distribution
    unique, counts = np.unique(results['labels'], return_counts=True)
    print("\nCluster distribution:")
    for cluster, count in zip(unique, counts):
        print(f"Cluster {cluster}: {count} sentences")
    
    # Print sample sentences from each cluster
    print("\nSample sentences from each cluster:")
    for cluster in unique:
        cluster_sentences = [s for s, l in zip(results['sentences'], results['labels']) if l == cluster]
        print(f"\nCluster {cluster} (sample):")
        for s in cluster_sentences[:3]:
            print(f"- {s}")