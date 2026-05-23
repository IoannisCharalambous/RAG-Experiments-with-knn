import numpy as np
import faiss
from annoy import AnnoyIndex
import os
import csv
import pickle
from sentence_transformers import SentenceTransformer
import time
import sys

# Reconfigure stdout to support UTF-8 on Windows to avoid UnicodeEncodeErrors
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# Define paths relative to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

csv_path = os.path.join(parent_dir, 'dbpedia_popular_os_regions.csv')
embeddings_path = os.path.join(script_dir, 'embeddings.npy')
ids_path = os.path.join(script_dir, 'keyword_ids.npy')
keywords_path = os.path.join(script_dir, 'keyword.txt')

faiss_output_csv = os.path.join(script_dir, 'knn_faiss_queries.csv')
annoy_output_csv = os.path.join(script_dir, 'ann_annoy_queries.csv')

annoy_index_file = os.path.join(script_dir, 'annoy_index.ann')
annoy_mapping_file = os.path.join(script_dir, 'annoy_id_map.pkl')

print("="*60)
print("🌍 DBpedia Popular Regions k-NN (FAISS) & ANN (Annoy) Benchmarker")
print("="*60)

# 1. Parse unique popular nodes from CSV
if not os.path.exists(csv_path):
    print(f"Error: CSV file not found at {csv_path}")
    exit(1)

print(f"Parsing popular lists from {csv_path}...")
popular_nodes = []
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if not row:
            continue
        node_name = row[0].strip()
        if node_name not in popular_nodes:
            popular_nodes.append(node_name)

print(f"Found {len(popular_nodes)} unique list entities in the CSV.")

# 2. Load dataset mappings
print("\nLoading dataset text mappings...")
id_to_text = {}
if os.path.exists(keywords_path):
    with open(keywords_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                id_to_text[int(parts[0])] = parts[1]
    print(f"Loaded {len(id_to_text):,} keyword mapping strings.")
else:
    print(f"Warning: {keywords_path} not found.")

# 3. Load embeddings
print("\nLoading embeddings and IDs (this can take a moment)...")
t0 = time.time()
embeddings = np.load(embeddings_path)
node_ids = np.load(ids_path)
dimension = embeddings.shape[1]
num_vectors = embeddings.shape[0]
print(f"Loaded {num_vectors:,} vectors of dimension {dimension} in {time.time() - t0:.2f}s")

# 4. Build/Load FAISS FlatL2 Index
print("\nBuilding FAISS FlatL2 index (exact k-NN)...")
t0 = time.time()
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)
print(f"FAISS index built in {time.time() - t0:.2f}s")

# 5. Build/Load Annoy Index
print("\nSetting up Annoy index (approximate ANN)...")
annoy_index = AnnoyIndex(dimension, 'euclidean')
id_mapping = {}

if os.path.exists(annoy_index_file) and os.path.exists(annoy_mapping_file):
    print("Found existing Annoy index & mapping file on disk. Loading...")
    t0 = time.time()
    annoy_index.load(annoy_index_file)
    with open(annoy_mapping_file, 'rb') as f:
        id_mapping = pickle.load(f)
    print(f"Annoy index & map loaded in {time.time() - t0:.2f}s")
else:
    print("Annoy index file not found. Building index (this can take a few minutes)...")
    t0 = time.time()
    
    print("Adding vectors to Annoy...")
    for i in range(num_vectors):
        annoy_index.add_item(i, embeddings[i])
        id_mapping[i] = node_ids[i]
        if i > 0 and i % 500000 == 0:
            print(f"  Added {i:,} vectors...")
            
    print("Building Annoy trees (using all cores)...")
    # n_trees = 50 balances build time and accuracy
    annoy_index.build(50, n_jobs=-1)
    print(f"Annoy index built in {time.time() - t0:.2f}s")
    
    print("Saving Annoy index & ID mapping to disk for future runs...")
    annoy_index.save(annoy_index_file)
    with open(annoy_mapping_file, 'wb') as f:
        pickle.dump(id_mapping, f)

# 6. Load SentenceTransformer model
print("\nLoading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

# 7. Generate target queries to evaluate
print("\nGenerating evaluation queries...")
queries_to_test = []

for node_name in popular_nodes:
    # Clean query
    clean_base = node_name.replace("List_of_", "").replace("_", " ")
    queries_to_test.append(clean_base)
    
    # Generate category/region variants
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

# Remove duplicates while preserving order
queries_to_test = list(dict.fromkeys(queries_to_test))
print(f"Generated {len(queries_to_test)} distinct semantic queries to test.")

# 8. Run searches and save to two CSV files
print(f"\nRunning queries on FAISS and writing to: {faiss_output_csv}...")
t_faiss_start = time.time()
with open(faiss_output_csv, 'w', newline='', encoding='utf-8') as f_faiss:
    writer = csv.writer(f_faiss)
    writer.writerow(["Query", "Rank", "Matched Keyword", "Matched ID", "Distance", "Search Time (ms)"])
    
    for idx, query in enumerate(queries_to_test):
        query_vector = model.encode([query], convert_to_numpy=True)
        
        # Profile only the FAISS index search lookup time
        t_q = time.time()
        distances, indices = faiss_index.search(query_vector, 20)
        q_time_ms = (time.time() - t_q) * 1000.0
        
        for rank in range(20):
            row_idx = indices[0][rank]
            matched_id = node_ids[row_idx]
            dist = distances[0][rank]
            matched_text = id_to_text.get(matched_id, f"ID {matched_id}")
            writer.writerow([query, rank + 1, matched_text, matched_id, f"{dist:.6f}", f"{q_time_ms:.3f}"])

faiss_duration = time.time() - t_faiss_start
print(f"FAISS queries completed in {faiss_duration:.2f}s")

print(f"\nRunning queries on Annoy and writing to: {annoy_output_csv}...")
t_annoy_start = time.time()
with open(annoy_output_csv, 'w', newline='', encoding='utf-8') as f_annoy:
    writer = csv.writer(f_annoy)
    writer.writerow(["Query", "Rank", "Matched Keyword", "Matched ID", "Distance", "Search Time (ms)"])
    
    for idx, query in enumerate(queries_to_test):
        query_vector = model.encode([query], convert_to_numpy=True)[0]
        
        # Profile only the Annoy index lookup time
        t_q = time.time()
        # search_k=50000 searches more trees/nodes for higher recall
        indices, distances = annoy_index.get_nns_by_vector(query_vector, 20, search_k=50000, include_distances=True)
        q_time_ms = (time.time() - t_q) * 1000.0
        
        for rank in range(20):
            ann_idx = indices[rank]
            matched_id = id_mapping[ann_idx]
            dist = distances[rank]
            matched_text = id_to_text.get(matched_id, f"ID {matched_id}")
            writer.writerow([query, rank + 1, matched_text, matched_id, f"{dist:.6f}", f"{q_time_ms:.3f}"])

annoy_duration = time.time() - t_annoy_start
print(f"Annoy queries completed in {annoy_duration:.2f}s")

print("\n" + "="*60)
print("SUCCESS: Completed dual k-NN and ANN query tests.")
print(f"FAISS Exact CSV: {faiss_output_csv}")
print(f"Annoy Approx CSV: {annoy_output_csv}")
print("="*60)
