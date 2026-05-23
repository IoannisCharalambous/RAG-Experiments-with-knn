import numpy as np
import hnswlib
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import time
import os
import pickle

# Define paths relative to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
embeddings_path = os.path.join(script_dir, 'embeddings.npy')
ids_path = os.path.join(script_dir, 'keyword_ids.npy')
keywords_path = os.path.join(script_dir, 'keyword.txt')

hnsw_index_file = os.path.join(script_dir, 'hnsw_index.bin')
annoy_index_file = os.path.join(script_dir, 'annoy_index.ann')
annoy_mapping_file = os.path.join(script_dir, 'annoy_id_map.pkl')

print("Loading dataset files...")
embeddings = np.load(embeddings_path)
node_ids = np.load(ids_path)
dimension = embeddings.shape[1]
num_elements = embeddings.shape[0]
print(f"Loaded {num_elements:,} vectors of dimension {dimension}")

# Load keyword texts
id_to_text = {}
if os.path.exists(keywords_path):
    with open(keywords_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                id_to_text[int(parts[0])] = parts[1]
else:
    print(f"Warning: {keywords_path} not found.")

# Load query model
print("\nLoading SentenceTransformer model for queries...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

query = "Best english Museums"
print(f"Query text: '{query}'")
query_vector = model.encode([query], convert_to_numpy=True)
query_vector_1d = query_vector[0]

# ----------------------------------------------------
# 1. HNSW Benchmark (via hnswlib)
# ----------------------------------------------------
print("\n" + "="*50)
print("1. RUNNING HNSW BENCHMARK")
print("="*50)

if os.path.exists(hnsw_index_file):
    print("Found existing HNSW index file on disk. Loading...")
    t0 = time.time()
    hnsw_index = hnswlib.Index(space='l2', dim=dimension)
    hnsw_index.load_index(hnsw_index_file, max_elements=num_elements)
    load_time = time.time() - t0
    print(f"HNSW index loaded from disk in: {load_time:.3f}s")
else:
    print("HNSW index file not found. Building index (this can take a few minutes)...")
    t0 = time.time()
    hnsw_index = hnswlib.Index(space='l2', dim=dimension)
    # ef_construction=150, M=16 are standard parameters balancing accuracy/build time
    hnsw_index.init_index(max_elements=num_elements, ef_construction=150, M=16)
    
    # Enable multi-threaded insertion
    hnsw_index.set_num_threads(os.cpu_count())
    hnsw_index.add_items(embeddings, node_ids)
    build_time = time.time() - t0
    print(f"HNSW index built in: {build_time:.3f}s")
    
    print("Saving HNSW index to disk...")
    hnsw_index.save_index(hnsw_index_file)

# Run Query on HNSW
hnsw_index.set_ef(50) # Search depth parameters
t0 = time.time()
labels, distances = hnsw_index.knn_query(query_vector, k=20)
hnsw_search_time = time.time() - t0
print(f"HNSW search execution time: {hnsw_search_time:.6f} seconds")

print("\nHNSW Top Results:")
for i in range(20):
    matched_id = labels[0][i]
    dist = distances[0][i]
    matched_text = id_to_text.get(matched_id, "Text not found")
    print(f"  Rank {i+1}: '{matched_text}' (ID: {matched_id}) | Distance: {dist:.4f}")


# ----------------------------------------------------
# 2. Annoy Benchmark (via annoy)
# ----------------------------------------------------
print("\n" + "="*50)
print("2. RUNNING ANNOY BENCHMARK")
print("="*50)

if os.path.exists(annoy_index_file) and os.path.exists(annoy_mapping_file):
    print("Found existing Annoy index & mapping file on disk. Loading...")
    t0 = time.time()
    annoy_index = AnnoyIndex(dimension, 'euclidean')
    annoy_index.load(annoy_index_file)
    with open(annoy_mapping_file, 'rb') as f:
        id_mapping = pickle.load(f)
    load_time = time.time() - t0
    print(f"Annoy index & map loaded in: {load_time:.3f}s")
else:
    print("Annoy index file not found. Building index (this can take a few minutes)...")
    t0 = time.time()
    annoy_index = AnnoyIndex(dimension, 'euclidean')
    
    id_mapping = {}
    print("Adding vectors to Annoy...")
    for i in range(num_elements):
        annoy_index.add_item(i, embeddings[i])
        id_mapping[i] = node_ids[i]
        if i > 0 and i % 500000 == 0:
            print(f"  Added {i:,} vectors...")
            
    print("Building Annoy trees (using all cores)...")
    # n_trees = 50 balances build time and accuracy
    annoy_index.build(50, n_jobs=-1)
    build_time = time.time() - t0
    print(f"Annoy index built in: {build_time:.3f}s")
    
    print("Saving Annoy index & ID mapping to disk...")
    annoy_index.save(annoy_index_file)
    with open(annoy_mapping_file, 'wb') as f:
        pickle.dump(id_mapping, f)

# Run Query on Annoy
t0 = time.time()
indices, distances = annoy_index.get_nns_by_vector(query_vector_1d, 20, search_k=-1, include_distances=True)
annoy_search_time = time.time() - t0
print(f"Annoy search execution time: {annoy_search_time:.6f} seconds")

print("\nAnnoy Top Results:")
for i in range(20):
    idx = indices[i]
    matched_id = id_mapping[idx]
    dist = distances[i]
    matched_text = id_to_text.get(matched_id, "Text not found")
    print(f"  Rank {i+1}: '{matched_text}' (ID: {matched_id}) | Distance: {dist:.4f}")

print("\n" + "="*50)
print("BENCHMARK SUMMARY")
print("="*50)
print(f"HNSW Search Latency  : {hnsw_search_time * 1000:.3f} ms")
print(f"Annoy Search Latency : {annoy_search_time * 1000:.3f} ms")
print("Index files generated:")
print(f"  - HNSW Index : {hnsw_index_file} ({os.path.getsize(hnsw_index_file)/(1024*1024):.1f} MB)")
print(f"  - Annoy Index: {annoy_index_file} ({os.path.getsize(annoy_index_file)/(1024*1024):.1f} MB)")
