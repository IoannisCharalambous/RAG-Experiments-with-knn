import os
import csv
import sys
import math

# Reconfigure stdout to support UTF-8 on Windows to avoid encoding errors
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
faiss_csv = os.path.join(script_dir, 'knn_faiss_queries.csv')
annoy_csv = os.path.join(script_dir, 'ann_annoy_queries.csv')
output_csv = os.path.join(script_dir, 'query_similarity_comparison.csv')
output_md = os.path.join(script_dir, 'query_similarity_comparison.md')

# Load and calculate metrics for each query
def load_metrics(file_path, is_faiss=False):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            if not row:
                continue
            query = row[0]
            distance = float(row[4])
            # Default to 0.0 if time column is missing
            q_time = float(row[5]) if len(row) > 5 else 0.0
            
            # Convert FAISS Squared L2 to standard L2 (Euclidean distance)
            if is_faiss:
                distance = math.sqrt(max(0.0, distance))
                
            if query not in data:
                data[query] = {'distance_sum': 0.0, 'time': q_time}
            data[query]['distance_sum'] += distance
    return data

if not os.path.exists(faiss_csv) or not os.path.exists(annoy_csv):
    print("Error: Please run test_knn_regions.py first to generate the input CSVs.")
    exit(1)

faiss_data = load_metrics(faiss_csv, is_faiss=True)
annoy_data = load_metrics(annoy_csv, is_faiss=False)

# Print table header
print("\n" + "-" * 115)
print(f"{'Query':<45} | {'FAISS L2':<9} | {'Annoy L2':<9} | {'% L2 Diff':<9} | {'FAISS Time':<10} | {'Annoy Time':<10} | {'Speedup':<8}")
print("-" * 115)

total_pct_diff = 0.0
total_faiss_time = 0.0
total_annoy_time = 0.0
count = 0

with open(output_csv, 'w', newline='', encoding='utf-8') as f_csv, \
     open(output_md, 'w', encoding='utf-8') as f_md:
     
    writer = csv.writer(f_csv)
    writer.writerow(["Query", "FAISS Sum Distance", "Annoy Sum Distance", "% Distance Diff", "FAISS Time (ms)", "Annoy Time (ms)", "Speedup"])
    
    # Write Markdown Header
    f_md.write("# FAISS vs. Annoy Query Quality & Performance Comparison\n\n")
    f_md.write("This table compares search accuracy (sum of standard L2 distances for the top 20 retrieved matches) and search speed (execution time in milliseconds) between FAISS (Exact k-NN) and Annoy (Approximate ANN).\n\n")
    f_md.write("| Query | FAISS L2 Sum | Annoy L2 Sum | % L2 Diff | FAISS Time (ms) | Annoy Time (ms) | Speedup |\n")
    f_md.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    
    for query in sorted(faiss_data.keys()):
        faiss_score = faiss_data[query]['distance_sum']
        faiss_time = faiss_data[query]['time']
        
        annoy_score = annoy_data.get(query, {'distance_sum': 0.0, 'time': 0.0})['distance_sum']
        annoy_time = annoy_data.get(query, {'distance_sum': 0.0, 'time': 0.0})['time']
        
        # Calculate percentage difference (distance inflation)
        if faiss_score > 0:
            pct_diff = ((annoy_score - faiss_score) / faiss_score) * 100.0
        else:
            pct_diff = 0.0
            
        # Calculate speedup (FAISS time / Annoy time)
        speedup = faiss_time / annoy_time if annoy_time > 0 else 1.0
        
        total_pct_diff += pct_diff
        total_faiss_time += faiss_time
        total_annoy_time += annoy_time
        count += 1
        
        # Console output
        print(f"{query:<45} | {faiss_score:<9.4f} | {annoy_score:<9.4f} | {pct_diff:<8.2f}% | {faiss_time:<8.3f}ms | {annoy_time:<8.3f}ms | {speedup:<6.2f}x")
        
        # Save to CSV
        writer.writerow([query, f"{faiss_score:.4f}", f"{annoy_score:.4f}", f"{pct_diff:.2f}%", f"{faiss_time:.3f}", f"{annoy_time:.3f}", f"{speedup:.2f}x"])
        
        # Save to MD
        f_md.write(f"| {query} | {faiss_score:.4f} | {annoy_score:.4f} | {pct_diff:.2f}% | {faiss_time:.3f} | {annoy_time:.3f} | {speedup:.2f}x |\n")

    # Global Statistics
    avg_pct_diff = total_pct_diff / count if count > 0 else 0.0
    avg_faiss_time = total_faiss_time / count if count > 0 else 0.0
    avg_annoy_time = total_annoy_time / count if count > 0 else 0.0
    global_speedup = avg_faiss_time / avg_annoy_time if avg_annoy_time > 0 else 1.0
    
    # Write summary statistics to MD
    f_md.write(f"\n## Summary Statistics\n")
    f_md.write(f"- **Average Distance Accuracy Loss (% Difference)**: **{avg_pct_diff:.2f}%**\n")
    f_md.write(f"- **Average FAISS Search Time**: **{avg_faiss_time:.3f} ms**\n")
    f_md.write(f"- **Average Annoy Search Time**: **{avg_annoy_time:.3f} ms**\n")
    f_md.write(f"- **Average Speedup Factor (FAISS/Annoy)**: **{global_speedup:.2f}x**\n")

print("-" * 115)
print(f"Summary: Avg L2 Loss: {avg_pct_diff:.2f}% | Avg FAISS Time: {avg_faiss_time:.3f}ms | Avg Annoy Time: {avg_annoy_time:.3f}ms | Speedup: {global_speedup:.2f}x")
print(f"Comparison results saved to CSV: {output_csv}")
print(f"Comparison results saved to MD:  {output_md}\n")
