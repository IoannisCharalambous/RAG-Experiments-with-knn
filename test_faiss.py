import numpy as np
import faiss
import time
import gc

def test_memory_limits():
    # ---------------------------------------------------
    # Setup Parameters
    # ---------------------------------------------------
    dimension = 128
    batch_size = 1000000 # 1 million vectors per batch
    max_batches = 15     # We will try to load up to 15 million vectors (~7.5 GB)
    
    # Calculate theoretical memory footprint per batch
    bytes_per_vector = dimension * 4 # float32 takes 4 bytes per number
    mb_per_batch = (batch_size * bytes_per_vector) / (1024 * 1024)
    
    print(f"Each batch of {batch_size:,} vectors takes ~{mb_per_batch:.2f} MB of memory.")
    print("WARNING: This test intentionally pushes your hardware until it fails!\n")

    # ---------------------------------------------------
    # 1. GPU Memory Stress Test
    # ---------------------------------------------------
    print("="*45)
    print("🚀 STARTING GPU MEMORY STRESS TEST (RTX 3050 Ti)")
    print("="*45)
    
    res = faiss.StandardGpuResources()
    # Create the index directly on the GPU this time
    gpu_index = faiss.GpuIndexFlatL2(res, dimension)
    gpu_vectors_added = 0
    
    try:
        for i in range(max_batches):
            # 1. Generate a batch
            batch = np.random.random((batch_size, dimension)).astype('float32')
            
            # 2. Try to add it to the GPU
            start_time = time.time()
            gpu_index.add(batch)
            add_time = time.time() - start_time
            
            gpu_vectors_added += batch_size
            print(f"✅ GPU holding {gpu_vectors_added:>10,} vectors | Add time: {add_time:.3f}s")
            
            # 3. Clean up the local batch to save system RAM
            del batch
            gc.collect()
            
    except Exception as e:
        print("\n❌ GPU OUT OF MEMORY!")
        print(f"Your RTX 3050 Ti hit its VRAM limit at {gpu_vectors_added:,} vectors.")
        # We catch the error so the script can continue to the CPU test
    
    # Free up the GPU entirely before starting the CPU test
    del gpu_index
    del res
    gc.collect()
    print("\nGPU memory cleared.\n")

    # ---------------------------------------------------
    # 2. CPU Memory Stress Test
    # ---------------------------------------------------
    print("="*45)
    print("🖥️ STARTING CPU MEMORY STRESS TEST")
    print("="*45)
    
    cpu_index = faiss.IndexFlatL2(dimension)
    cpu_vectors_added = 0
    
    try:
        for i in range(max_batches):
            batch = np.random.random((batch_size, dimension)).astype('float32')
            
            start_time = time.time()
            cpu_index.add(batch)
            add_time = time.time() - start_time
            
            cpu_vectors_added += batch_size
            print(f"✅ CPU holding {cpu_vectors_added:>10,} vectors | Add time: {add_time:.3f}s")
            
            del batch
            gc.collect()
            
        print("\n✅ CPU successfully held all 15,000,000 vectors without crashing!")
        print("Your System RAM is much larger than your GPU VRAM.")
        
    except MemoryError:
         print("\n❌ CPU OUT OF MEMORY!")
         print(f"Your System RAM hit its limit at {cpu_vectors_added:,} vectors.")
    except Exception as e:
         print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_memory_limits()