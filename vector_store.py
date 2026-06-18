import chromadb

# Create a local ChromaDB client — stores data on your machine
client = chromadb.Client()

def store_chunks(chunks, embeddings):
    # Delete collection if it already exists (for when user uploads a new PDF)
    existing = [c.name for c in client.list_collections()]
    if "research_papers" in existing:
        client.delete_collection("research_papers")
    
    # Create fresh collection
    collection = client.create_collection("research_papers")
    
    # Add chunks with their embeddings
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[str(i) for i in range(len(chunks))]
    )
    
    return collection


def search_chunks(collection, query_embedding, n_results=7):
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    
    # Return just the text chunks, not the metadata
    return results["documents"][0]