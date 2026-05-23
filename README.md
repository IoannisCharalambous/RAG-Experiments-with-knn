# RAG Nearest Neighbor Search & Benchmarking Project

This repository is dedicated to the research, implementation, and benchmarking of vector search indexes for **Retrieval-Augmented Generation (RAG)** systems. It processes millions of DBpedia keywords using sentence transformer embeddings, evaluates exact k-NN vs. Approximate Nearest Neighbors (ANN) indexes, and profiles system hardware boundaries.

---

## 📂 Repository Directory Structure

```text
├── dbpedia_popular_os_regions.csv        # Geographical list entities and density regions from DBpedia
├── FAISS_kNN.docx                        # Notes/theory document on FAISS nearest-neighbor searches
├── test_faiss.py                         # GPU/CPU memory stress test profiling script
├── README.md                             # Repository summary (this file)
└── exploration with fasiss/              # Main workspace folder containing profiling and index code
    ├── embeddings.py                     # Script generating vector embeddings from text corpus
    ├── test_knn_regions.py               # Dual execution pipeline generating queries and search outputs
    ├── compare_search_quality.py         # Comparative analysis script for FAISS vs. Annoy
    ├── ann_alternatives.py               # Script comparing CPU exact FAISS to FAISS Inverted File (IVF) index
    ├── kNN_propartionality_faiss.py      # Script exploring results diversity and proportionality scores
    ├── nbaiot.py                         # Dataset builder script for the N-BaIoT IoT intrusion dataset
    ├── embeddings.npy                    # [Serialized] 4.5 GB matrix of 2.9 million sentence embeddings
    ├── keyword_ids.npy                   # [Serialized] Mapping IDs for the embedded texts
    ├── keyword.txt                       # Raw text DBpedia corpus (ID -> text keyword mapping)
    ├── hnsw_index.bin                    # [Cached] Pre-built HNSW graph index (FAISS, ~4.9 GB)
    ├── annoy_index.ann                   # [Cached] Pre-built Annoy trees index (~4.5 GB)
    ├── annoy_id_map.pkl                  # [Cached] Serialized map dictionary for Annoy integer-to-DBpedia IDs
    ├── knn_faiss_queries.csv             # Output results for 52 queries using FAISS Exact Search
    ├── ann_annoy_queries.csv             # Output results for 52 queries using Annoy Approximate Search
    ├── query_similarity_comparison.csv   # Tabular comparison of similarity sums and query latency
    └── query_similarity_comparison.md    # Markdown report displaying similarity scores and speed comparison
```

---

## 💾 Core Datasets & Indexes

* **Text Corpus (`keyword.txt`)**: A dataset consisting of **2,927,026** lowercase, normalized keywords and entity tags from DBpedia.
* **Vector Embeddings (`embeddings.npy`)**: A **4.5 GB** dense matrix storing embeddings ($2,927,026 \times 384$) generated via the `all-MiniLM-L6-v2` transformer model.
* **HNSW Index (`hnsw_index.bin`)**: A precompiled Hierarchical Navigable Small World graph index for FAISS, occupying **4.9 GB** on disk, permitting sub-millisecond searches.
* **Annoy Index (`annoy_index.ann`)**: A precompiled approximate search index utilizing **50 random projection trees** that supports memory mapping (`mmap`), allowing searches directly from disk with minimal memory overhead.

---

## ⚙️ Core Scripts & Workflows

### 1. Vector Search Performance Profiling
* **[test_faiss.py](file:///d:/Users/ioann/Documents/uni/Internship/Fakas/RAG/test_faiss.py)**: Intentionally stress-tests memory capacity by loading batches of random 128-dimensional vectors to find the system RAM limits and Laptop GPU VRAM limits (specifically targetting a Laptop GeForce RTX 3050 Ti VRAM threshold).

### 2. Search Pipelines Comparison
* **[exploration with fasiss/test_knn_regions.py](file:///d:/Users/ioann/Documents/uni/Internship/Fakas/RAG/exploration%20with%20fasiss/test_knn_regions.py)**:
  * Automatically reads unique category lists from `dbpedia_popular_os_regions.csv`.
  * Generates **52 semantic and geographic query strings** (e.g. `"Harvard alumni in Massachusetts"`, `"2010 FIFA World Cup squads"`).
  * Executes exact search in FAISS (`IndexFlatL2`) and approximate search in Annoy (`AnnoyIndex`), logging the top 20 matches, distances, and search latency (in milliseconds) into separate CSV outputs.

### 3. Metric Calculations & Speedups
* **[exploration with fasiss/compare_search_quality.py](file:///d:/Users/ioann/Documents/uni/Internship/Fakas/RAG/exploration%20with%20fasiss/compare_search_quality.py)**:
  * Reads generated outputs and corrects the distance formulas (taking the square root of FAISS squared L2 values to match Annoy's standard L2 distances).
  * Calculates the sum of distances for top 20 results (evaluating loss) and speedup factors for all queries.
  * Outputs the final comparison matrix to `query_similarity_comparison.csv` and a Markdown summary table.

---

## 📈 Key Benchmark Insights

Using the evaluation dataset of 2,927,026 vectors (384 dimensions) on an AMD/Intel multi-core CPU:

1. **Massive Speedups**: Approximate search via Annoy (`search_k=50000`) completes search queries in **`46.61 ms`** on average, compared to **`279.83 ms`** for exact FAISS linear scanning. This represents an average **`6.00x` speedup factor**.
2. **Minimal Accuracy Loss**: The average distance accuracy loss (% difference in L2 distance sums) when moving from exact search to Annoy is **only `0.85%`**, confirming that approximate search retains near-perfect semantic mapping.
3. **Index Caching**: Compiling and serializing the indices to disk saves valuable CPU cycles. Loading pre-built graphs from disk takes **`< 1.5s`**, compared to building them from scratch which takes several minutes.
