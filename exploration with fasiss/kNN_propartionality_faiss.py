import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import os
# Configure your Hugging Face token here or set it as an environment variable HF_TOKEN
# os.environ["HF_TOKEN"] = "your_token_here"

embeddings = np.load('embeddings.npy')
node_ids = np.load('keyword_ids.npy')
dimension = embeddings.shape[1]

res = faiss.StandardGpuResources()
cpu_index = faiss.IndexFlatL2(dimension)

try:
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.add(embeddings)
    print(f"FAISS GPU index ready! Total vectors: {gpu_index.ntotal}")
except RuntimeError as e:
    print("\nOut of VRAM.")
    print("Error details:", e)
    exit()

print("Loading SentenceTransformer model for queries...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

id_to_text = {}
with open('keyword.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            id_to_text[int(parts[0])] = parts[1]

print("Ready to search!\n" + "-"*50)

query = "Best english Museums"
print(f"Searching for matches to: '{query}'")

query_vector = model.encode([query], convert_to_numpy=True)
k = 20
distances, result_row_indices = gpu_index.search(query_vector, k)

top_k_indices = result_row_indices[0]
top_k_embeddings = embeddings[top_k_indices]

# Calculate Euclidean distances among top k results (found this api online, not sure if it's correct)
pairwise_distances = cdist(top_k_embeddings, top_k_embeddings, metric='euclidean')

max_D = np.max(pairwise_distances)

if max_D == 0:
    max_D = 1e-9 

s = 1 - (pairwise_distances / max_D)

np.fill_diagonal(s, 0)


sR_scores = np.sum(s, axis=1)

print("\nTop Matches with Proportionality Scores:")
for i in range(k):
    row_index = top_k_indices[i]
    matched_id = node_ids[row_index]
    
    faiss_distance = distances[0][i]
    proportionality = sR_scores[i]
    matched_text = id_to_text.get(matched_id, "Text not found")
    
    print(f"Rank {i+1}: '{matched_text}' (ID: {matched_id})")
    print(f"FAISS Distance: {faiss_distance:.4f}")
    print(f"Proportionality: {proportionality:.4f}\n")