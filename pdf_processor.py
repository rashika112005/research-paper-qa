import fitz  # PyMuPDF

def extract_text_from_pdf(uploaded_file):
    # Open the PDF from the uploaded file bytes
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    return full_text


def split_into_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks