import faiss
import numpy as np

# Global storage
chunk_store = []
index = None

def store_chunks(chunks, embeddings):
    global chunk_store, index
    
    chunk_store = chunks
    embeddings_np = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    
    return index

def search_chunks(collection, query_embedding, n_results=7):
    global chunk_store
    
    query_np = np.array([query_embedding]).astype('float32')
    distances, indices = collection.search(query_np, n_results)
    
    results = [chunk_store[i] for i in indices[0] if i < len(chunk_store)]
    return results