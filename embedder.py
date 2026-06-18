from sentence_transformers import SentenceTransformer

# Load the model once — this downloads it first time, then uses cached version
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings

def get_query_embedding(query):
    # Map common question types to better search terms
    query_lower = query.lower()
    
    if "contribution" in query_lower or "propose" in query_lower:
        enhanced_query = "our contributions are novel proposed method summary"
    elif "dann" in query_lower or "compare" in query_lower or "comparison" in query_lower:
        enhanced_query = "DANN comparison mAP AID UCM results table"
    elif "dataset" in query_lower:
        enhanced_query = "datasets used experiments benchmark"
    else:
        enhanced_query = query
        
    embedding = model.encode(enhanced_query)
    return embedding