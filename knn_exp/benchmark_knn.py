import numpy as np
import faiss
from annoy import AnnoyIndex
import os
import csv
import pickle
from sentence_transformers import SentenceTransformer
import time
import sys
import math

class DataManager:
    def __init__(self, base_dir):
        self.dataset_dir = os.path.join(base_dir, 'datasets', 'dbpedia', 'embeddings')
        self.csv_path = os.path.join(self.dataset_dir, 'dbpedia_popular_os_regions.csv')
        self.embeddings_path = os.path.join(self.dataset_dir, 'embeddings.npy')
        self.ids_path = os.path.join(self.dataset_dir, 'keyword_ids.npy')
        self.keywords_path = os.path.join(self.dataset_dir, 'keyword.txt')
        self.annoy_index_file = os.path.join(self.dataset_dir, 'annoy_index.ann')
        self.annoy_mapping_file = os.path.join(self.dataset_dir, 'annoy_id_map.pkl')
        
    def load_queries(self):
        popular_nodes = []
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row:
                    node_name = row[0].strip()
                    if node_name not in popular_nodes:
                        popular_nodes.append(node_name)
        
        queries_to_test = []
        for node_name in popular_nodes:
            clean_base = node_name.replace("List_of_", "").replace("_", " ")
            queries_to_test.append(clean_base)
            if "World_Cup" in node_name:
                year = node_name.split("_")[0]
                queries_to_test.append(f"{year} football tournament players")
                queries_to_test.append(f"FIFA football cup squads from {year}")
            elif "Harvard" in node_name:
                queries_to_test.append("Harvard alumni in Massachusetts")
                queries_to_test.append("famous Harvard University professors")
            elif "chapters" in node_name:
                frat_name = clean_base.replace(" chapters", "")
                queries_to_test.append(f"{frat_name} college fraternity chapters")
                queries_to_test.append(f"American college fraternity {frat_name}")
            elif "Basketball" in node_name:
                queries_to_test.append("NCAA basketball tournament brackets")
            elif "Fighting_Illini" in node_name:
                queries_to_test.append("Illinois college basketball team players")
        
        return list(dict.fromkeys(queries_to_test))
        
    def load_id_to_text(self):
        id_to_text = {}
        if os.path.exists(self.keywords_path):
            with open(self.keywords_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        id_to_text[int(parts[0])] = parts[1]
        return id_to_text
        
    def load_embeddings_and_ids(self):
        # Using mmap to avoid loading the entire 4.5GB array into RAM at once
        embeddings = np.load(self.embeddings_path, mmap_mode='r')
        node_ids = np.load(self.ids_path)
        return embeddings, node_ids

class Evaluator:
    def __init__(self, results_dir, id_to_text):
        self.results_dir = results_dir
        self.id_to_text = id_to_text
        self.metrics = {} # {algo: {query: {'dist': x, 'time': y}}}
        
    def write_results_to_csv(self, algo_name, queries, results):
        csv_path = os.path.join(self.results_dir, f'{algo_name}_results.csv')
        self.metrics[algo_name] = {}
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Query", "Rank", "Matched Keyword", "Matched ID", "Distance", "Search Time (ms)"])
            
            for q_idx, query in enumerate(queries):
                matched_ids, distances, q_time_ms = results[q_idx]
                
                sum_dist = 0
                for rank, (matched_id, dist) in enumerate(zip(matched_ids, distances)):
                    if matched_id == -1: 
                        continue
                    matched_text = self.id_to_text.get(matched_id, f"ID {matched_id}")
                    writer.writerow([query, rank + 1, matched_text, matched_id, f"{dist:.6f}", f"{q_time_ms:.3f}"])
                    sum_dist += dist
                    
                self.metrics[algo_name][query] = {'distance_sum': sum_dist, 'time': q_time_ms}

    def generate_summary(self, queries):
        md_path = os.path.join(self.results_dir, 'comparison_summary.md')
        
        base_algo = 'faiss_l2'
        if base_algo not in self.metrics:
            return
            
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# KNN Algorithms Comparison Summary\n\n")
            
            algos = list(self.metrics.keys())
            
            # Write header
            f.write("| Query | ")
            for algo in algos:
                f.write(f"{algo} L2 | {algo} Time (ms) | ")
                if algo != base_algo:
                    f.write(f"% L2 Diff | Speedup | ")
            f.write("\n| :--- | ")
            for algo in algos:
                f.write(":---: | :---: | ")
                if algo != base_algo:
                    f.write(":---: | :---: | ")
            f.write("\n")
            
            global_stats = {algo: {'pct_diff': 0, 'time': 0, 'count': 0} for algo in algos}
            
            for query in queries:
                f.write(f"| {query} | ")
                base_dist = self.metrics[base_algo][query]['distance_sum']
                base_time = self.metrics[base_algo][query]['time']
                
                for algo in algos:
                    dist = self.metrics[algo][query]['distance_sum']
                    time_ms = self.metrics[algo][query]['time']
                    f.write(f"{dist:.4f} | {time_ms:.3f} | ")
                    
                    global_stats[algo]['time'] += time_ms
                    global_stats[algo]['count'] += 1
                    
                    if algo != base_algo:
                        pct_diff = ((dist - base_dist) / base_dist * 100) if base_dist > 0 else 0
                        speedup = base_time / time_ms if time_ms > 0 else 1
                        f.write(f"{pct_diff:.2f}% | {speedup:.2f}x | ")
                        global_stats[algo]['pct_diff'] += pct_diff
                f.write("\n")
                
            f.write("\n## Summary Statistics\n")
            base_avg_time = global_stats[base_algo]['time'] / global_stats[base_algo]['count']
            f.write(f"- **{base_algo} Average Time**: {base_avg_time:.3f} ms\n")
            
            for algo in algos:
                if algo == base_algo: continue
                avg_pct_diff = global_stats[algo]['pct_diff'] / global_stats[algo]['count']
                avg_time = global_stats[algo]['time'] / global_stats[algo]['count']
                speedup = base_avg_time / avg_time if avg_time > 0 else 1
                f.write(f"- **{algo} Average Time**: {avg_time:.3f} ms (Speedup: {speedup:.2f}x)\n")
                f.write(f"- **{algo} Average Accuracy Loss (% Diff)**: {avg_pct_diff:.2f}%\n")


def run_faiss_ivf(embeddings, query_vectors, node_ids):
    num_vectors, dimension = embeddings.shape
    nlist = int(4 * np.sqrt(num_vectors))
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
    
    # Train
    train_size = min(100000, num_vectors)
    train_indices = np.random.choice(num_vectors, train_size, replace=False)
    # Read training data explicitly to pass to FAISS
    train_data = np.array([embeddings[i] for i in train_indices]) 
    index.train(train_data)
    
    # Add vectors in chunks to prevent large RAM spikes with mmap arrays
    chunk_size = 500000
    for i in range(0, num_vectors, chunk_size):
        end = min(i + chunk_size, num_vectors)
        index.add(np.array(embeddings[i:end]))
        
    index.nprobe = 10
    results = []
    for q_vec in query_vectors:
        t0 = time.time()
        distances, indices = index.search(np.array([q_vec]), 20)
        q_time_ms = (time.time() - t0) * 1000
        euclidean_distances = [math.sqrt(max(0, d)) for d in distances[0]]
        matched_ids = [node_ids[i] if i != -1 else -1 for i in indices[0]]
        results.append((matched_ids, euclidean_distances, q_time_ms))
    return results

def run_annoy(embeddings, query_vectors, node_ids, annoy_index_path, annoy_map_path):
    dimension = embeddings.shape[1]
    num_vectors = embeddings.shape[0]
    index = AnnoyIndex(dimension, 'euclidean')
    
    id_mapping = {}
    if os.path.exists(annoy_index_path) and os.path.exists(annoy_map_path):
        print("Loading existing Annoy index...")
        index.load(annoy_index_path)
        with open(annoy_map_path, 'rb') as f:
            id_mapping = pickle.load(f)
    else:
        print("Building Annoy index (this may take a while)...")
        for i in range(num_vectors):
            index.add_item(i, embeddings[i])
            id_mapping[i] = node_ids[i]
            if i > 0 and i % 500000 == 0:
                print(f"  Added {i:,} vectors...")
        index.build(50, n_jobs=-1)
        index.save(annoy_index_path)
        with open(annoy_map_path, 'wb') as f:
            pickle.dump(id_mapping, f)
            
    results = []
    for q_vec in query_vectors:
        t0 = time.time()
        indices, distances = index.get_nns_by_vector(q_vec, 20, search_k=50000, include_distances=True)
        q_time_ms = (time.time() - t0) * 1000
        matched_ids = [id_mapping[i] for i in indices]
        results.append((matched_ids, distances, q_time_ms))
    return results

def main():
    if sys.platform.startswith('win'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    dm = DataManager(base_dir)
    print("Loading queries...")
    queries = dm.load_queries()
    print(f"Generated {len(queries)} queries.")
    
    print("Loading id to text mapping...")
    id_to_text = dm.load_id_to_text()
    
    print("Loading embeddings and ids...")
    embeddings, node_ids = dm.load_embeddings_and_ids()
    print(f"Embeddings shape: {embeddings.shape}")
    
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if faiss.get_num_gpus() > 0 else 'cpu')
    query_vectors = model.encode(queries, convert_to_numpy=True)
    
    evaluator = Evaluator(script_dir, id_to_text)
    
    print("\nRunning FAISS L2 Exact...")
    dimension = embeddings.shape[1]
    num_vectors = embeddings.shape[0]
    l2_index = faiss.IndexFlatL2(dimension)
    chunk_size = 500000
    for i in range(0, num_vectors, chunk_size):
        end = min(i + chunk_size, num_vectors)
        l2_index.add(np.array(embeddings[i:end]))
        
    l2_results = []
    for q_vec in query_vectors:
        t0 = time.time()
        distances, indices = l2_index.search(np.array([q_vec]), 20)
        q_time_ms = (time.time() - t0) * 1000
        euclidean_distances = [math.sqrt(max(0, d)) for d in distances[0]]
        matched_ids = [node_ids[i] for i in indices[0]]
        l2_results.append((matched_ids, euclidean_distances, q_time_ms))
        
    evaluator.write_results_to_csv('faiss_l2', queries, l2_results)
    
    print("\nRunning FAISS IVF...")
    ivf_results = run_faiss_ivf(embeddings, query_vectors, node_ids)
    evaluator.write_results_to_csv('faiss_ivf', queries, ivf_results)
    
    print("\nRunning Annoy...")
    annoy_results = run_annoy(embeddings, query_vectors, node_ids, dm.annoy_index_file, dm.annoy_mapping_file)
    evaluator.write_results_to_csv('annoy', queries, annoy_results)
    
    print("\nGenerating summary...")
    evaluator.generate_summary(queries)
    print("Done! Check knn_exp directory for results.")

if __name__ == "__main__":
    main()
