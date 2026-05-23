import numpy as np
from sentence_transformers import SentenceTransformer
import torch

if torch.cuda.is_available():
    print(f"GPU Detected! Using: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU detected. Falling back to CPU.")

print("Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

input_file = 'keyword.txt'
embeddings_output_file = 'embeddings.npy'
ids_output_file = 'keyword_ids.npy'

BATCH_SIZE = 100000
keyword_ids_batch = []
keyword_texts_batch = []

all_embeddings = []
all_keyword_ids = []
total_processed = 0

print(f"Starting to read from {input_file}...")

with open(input_file, 'r', encoding='utf-8') as infile:
    for line in infile:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            keyword_ids_batch.append(int(parts[0]))
            keyword_texts_batch.append(parts[1])
            
        if len(keyword_texts_batch) == BATCH_SIZE:
            
            embeddings = model.encode(
                keyword_texts_batch, 
                batch_size=1024, 
                convert_to_numpy=True, 
                show_progress_bar=True
            )
            
            all_embeddings.append(embeddings)
            all_keyword_ids.append(np.array(keyword_ids_batch, dtype=np.int64))
            
            total_processed += len(keyword_texts_batch)
            print(f"Processed {total_processed} keywords...")
            
            keyword_ids_batch = []
            keyword_texts_batch = []

    if keyword_texts_batch:
        embeddings = model.encode(
            keyword_texts_batch, 
            batch_size=1024, 
            convert_to_numpy=True
        )
        
        all_embeddings.append(embeddings)
        all_keyword_ids.append(np.array(keyword_ids_batch, dtype=np.int64))
        
        total_processed += len(keyword_texts_batch)
        print(f"Processed {total_processed} keywords...")

print("Stacking all batches into final arrays...")
final_embeddings_matrix = np.vstack(all_embeddings)
final_ids_array = np.concatenate(all_keyword_ids)

print("Saving to .npy files...")
np.save(embeddings_output_file, final_embeddings_matrix)
np.save(ids_output_file, final_ids_array)

print(f"\nSuccess! Saved {final_embeddings_matrix.shape[0]} vectors.")