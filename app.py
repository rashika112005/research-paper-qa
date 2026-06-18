import streamlit as st
from pdf_processor import extract_text_from_pdf, split_into_chunks
from embedder import get_embeddings
from vector_store import store_chunks, search_chunks
from llm import get_answer
from embedder import get_embeddings, get_query_embedding
import time

# Page config
st.set_page_config(page_title="ResearchChat", page_icon="📄")
st.title("📄 ResearchChat")
st.caption("Upload a research paper and ask questions about it!")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = None

# Sidebar — PDF upload
with st.sidebar:
    st.header("Upload Paper")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            # Step 1 - Extract text
            text = extract_text_from_pdf(uploaded_file)
            
            # Step 2 - Split into chunks
            chunks = split_into_chunks(text)
            
            # Step 3 - Get embeddings
            embeddings = get_embeddings(chunks)
            
            # Step 4 - Store in ChromaDB
            st.session_state.collection = store_chunks(chunks, embeddings)
        
        st.success(f"✅ Paper processed! {len(chunks)} chunks created.")
        st.info("Now ask any question about the paper!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_question = st.chat_input("Ask something about the paper...")

if user_question:
    # Check if PDF is uploaded
    if st.session_state.collection is None:
        st.warning("⚠️ Please upload a research paper first!")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    question_embedding = get_query_embedding(user_question)
                    relevant_chunks = search_chunks(
                        st.session_state.collection,
                        question_embedding
                    )
                    time.sleep(3)
                    answer = get_answer(user_question, relevant_chunks)
                    
                    st.write(answer)
                except Exception as e:
                    answer = f"⚠️ Error: {str(e)}"
                    st.warning(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})