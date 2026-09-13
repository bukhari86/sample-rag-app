import streamlit as st
from dotenv import load_dotenv
import os
from anthropic import Anthropic
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key)

# RAG functions
def chunk_by_word_count(text, chunk_size=100, overlap=10):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
    return chunks

def get_chunk_embedding(chunk, model):
    words = simple_preprocess(chunk)
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)

def search_chunks(question, chunks, model, top_k=2):
    question_embedding = get_chunk_embedding(question, model)
    similarities = []
    for i, chunk in enumerate(chunks):
        chunk_embedding = get_chunk_embedding(chunk, model)
        sim = cosine_similarity(question_embedding, chunk_embedding)
        similarities.append((i, chunk, sim))
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:top_k]

def answer_question_about_documents(document_text, question):
    chunks = chunk_by_word_count(document_text, chunk_size=50, overlap=5)
    sentences = [simple_preprocess(chunk) for chunk in chunks]
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1)
    results = search_chunks(question, chunks, model, top_k=2)
    retrieved_chunks = [chunk for _, chunk, _ in results]
    context = "\n\n".join(retrieved_chunks)
    
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=500,
        system="You are a helpful assistant. Answer the user's question based ONLY on the provided documents.",
        messages=[
            {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {question}"}
        ]
    )
    
    return response.content[0].text

# Streamlit UI
st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("📚 Document Question & Answer")
st.write("Upload a document and ask questions about it using AI.")

# Sidebar for document upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a text file", type="txt")
    
    if uploaded_file:
        document_text = uploaded_file.read().decode("utf-8")
        st.success("Document uploaded!")
        st.write(f"Document size: {len(document_text)} characters")

# Main area for questions
if uploaded_file:
    st.header("Ask Questions")
    question = st.text_input("Enter your question:")
    
    if question:
        st.write("Searching your document...")
        answer = answer_question_about_documents(document_text, question)
        
        st.subheader("Answer:")
        st.write(answer)
else:
    st.info("👈 Upload a document in the sidebar to get started")