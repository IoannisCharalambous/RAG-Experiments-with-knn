import numpy as np
import faiss
import time
import os
from sentence_transformers import SentenceTransformer

# Define paths relative to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
embeddings_path = os.path.join(script_dir, 'embeddings.npy')
ids_path = os.path.join(script_dir, 'keyword_ids.npy')
keywords_path = os.path.join(script_dir, 'keyword.txt')

print("Loading dataset files...")
embeddings = np.load(embeddings_path)
node_ids = np.load(ids_path)
dimension = embeddings.shape[1]
num_vectors = embeddings.shape[0]
print(f"Loaded {num_vectors:,} vectors of dimension {dimension}")

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
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if faiss.get_num_gpus() > 0 else 'cpu')

# Set up test query
query = "Space exploration and lunar landing missions"
print(f"\nQuerying: '{query}'")
query_vector = model.encode([query], convert_to_numpy=True)

# ----------------------------------------------------
# 1. Exact kNN Search (FAISS IndexFlatL2) on CPU
# ----------------------------------------------------
print("\n" + "="*50)
print("1. Running EXACT Search (FAISS IndexFlatL2 on CPU)")
print("="*50)

t0 = time.time()
flat_index = faiss.IndexFlatL2(dimension)
flat_index.add(embeddings)
index_time = time.time() - t0
print(f"Flat index built in: {index_time:.3f}s")

t0 = time.time()
distances, indices = flat_index.search(query_vector, 10)
search_time = time.time() - t0
print(f"Flat search time: {search_time:.4f}s")

print("Top 5 Results:")
for i in range(5):
    idx = indices[0][i]
    matched_id = node_ids[idx]
    text = id_to_text.get(matched_id, f"ID {matched_id}")
    print(f"  Rank {i+1}: '{text}' (Dist: {distances[0][i]:.4f})")

# ----------------------------------------------------
# 2. Approximate IVF Search (FAISS IndexIVFFlat on CPU)
# ----------------------------------------------------
print("\n" + "="*50)
print("2. Running APPROXIMATE Search (FAISS IndexIVFFlat on CPU)")
print("="*50)
print("Building an Inverted File (IVF) index splits the space into clusters (voronoi cells).")
print("Instead of scanning all vectors, it only scans the nearest clusters.")

# Rule of thumb: number of centroids (nlist) is around 4 * sqrt(N) to 16 * sqrt(N)
nlist = int(4 * np.sqrt(num_vectors)) # For ~2.9M vectors, this is ~6800 centroids
print(f"Using nlist = {nlist} clusters")

t0 = time.time()
quantizer = faiss.IndexFlatL2(dimension)
ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)

# IVF index needs training to determine cluster centroids
# We train it on a random subset of 100k vectors to save time
print("Training IVF centroids (this clustering takes a few seconds)...")
train_size = min(100000, num_vectors)
train_indices = np.random.choice(num_vectors, train_size, replace=False)
ivf_index.train(embeddings[train_indices])

# Add all vectors to the index
print("Adding vectors to IVF partitions...")
ivf_index.add(embeddings)
index_time = time.time() - t0
print(f"IVF index built & trained in: {index_time:.3f}s")

# Set nprobe (number of clusters to search). Higher = more accurate but slower.
# nprobe = 1 is fastest. nprobe = 10 to 64 is typical.
for nprobe in [1, 10, 50]:
    ivf_index.nprobe = nprobe
    t0 = time.time()
    distances_ivf, indices_ivf = ivf_index.search(query_vector, 10)
    search_time_ivf = time.time() - t0
    
    # Calculate recall relative to Exact Flat search
    exact_set = set(indices[0])
    approx_set = set(indices_ivf[0])
    recall = len(exact_set.intersection(approx_set)) / len(exact_set)
    
    print(f"\n[nprobe = {nprobe}] Search time: {search_time_ivf:.4f}s | Recall vs Exact: {recall:.1%}")
    print("Top 3 Results:")
    for i in range(3):
        idx = indices_ivf[0][i]
        if idx < 0:
            continue
        matched_id = node_ids[idx]
        text = id_to_text.get(matched_id, f"ID {matched_id}")
        print(f"  Rank {i+1}: '{text}' (Dist: {distances_ivf[0][i]:.4f})")

print("\nReady! You can run this script to compare exact and approximate FAISS search on your CPU.")
