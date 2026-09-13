import streamlit as st
from dotenv import load_dotenv
import os
from anthropic import Anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key)

def chunk_by_word_count(text, chunk_size=100, overlap=10):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
    return chunks

def simple_search(question, chunks):
    """Simple keyword-based search instead of embeddings."""
    question_words = set(question.lower().split())
    scored_chunks = []
    
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(question_words & chunk_words)
        scored_chunks.append((chunk, score))
    
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in scored_chunks[:2]]

def answer_question_about_documents(document_text, question):
    chunks = chunk_by_word_count(document_text, chunk_size=50, overlap=5)
    retrieved_chunks = simple_search(question, chunks)
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