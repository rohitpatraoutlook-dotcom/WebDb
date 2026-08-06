import os
import tempfile
import chromadb
from chromadb.utils import embedding_functions
import pymupdf4llm
import docx

# Initialize ChromaDB Local Client & Embedding Model
client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def extract_markdown_from_file(file_bytes, filename):
    """File bytes se markdown ya text extract karta hai"""
    if filename.endswith('.pdf'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name
        try:
            raw_markdown = pymupdf4llm.to_markdown(tmp_path)
        finally:
            os.remove(tmp_path)
            
    elif filename.endswith('.docx'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name
        try:
            doc = docx.Document(tmp_path)
            raw_markdown = "\n".join([p.text for p in doc.paragraphs])
        finally:
            os.remove(tmp_path)
            
    elif filename.endswith('.txt'):
        raw_markdown = file_bytes.decode('utf-8')
    else:
        raise ValueError("Unsupported file format")
        
    return raw_markdown

def split_text_into_chunks(text, chunk_size=600, overlap=60):
    """Text ko chhote chunks me divide karta hai"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def process_and_store_file(file_bytes, filename, collection_name):
    """Extraction, Chunking, aur ChromaDB Storage process"""
    raw_markdown = extract_markdown_from_file(file_bytes, filename)
    
    if not raw_markdown.strip():
        raise ValueError("File is empty or text could not be extracted")
        
    chunks = split_text_into_chunks(raw_markdown)
    
    collection = client.get_or_create_collection(
        name=collection_name, 
        embedding_function=embedding_fn
    )
    
    documents = chunks
    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"filename": filename, "chunk_id": i} for i in range(len(chunks))]
    
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    
    return len(chunks)

