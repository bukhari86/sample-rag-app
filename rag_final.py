from dotenv import load_dotenv
import os
from anthropic import Anthropic
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key)

def chunk_by_word_count(text, chunk_size=100, overlap=10):
    """Break text into chunks of roughly chunk_size words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
    return chunks

def get_chunk_embedding(chunk, model):
    """Convert chunk to embedding by averaging word vectors."""
    words = simple_preprocess(chunk)
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)

def search_chunks(question, chunks, model, top_k=2):
    """Find most relevant chunks for a question."""
    question_embedding = get_chunk_embedding(question, model)
    similarities = []
    for i, chunk in enumerate(chunks):
        chunk_embedding = get_chunk_embedding(chunk, model)
        sim = cosine_similarity(question_embedding, chunk_embedding)
        similarities.append((i, chunk, sim))
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:top_k]

def answer_question_about_documents(document_text, question):
    """
    Main function: Answer a question based on a document.
    
    Args:
        document_text: The full document to search through
        question: The question to answer
    
    Returns:
        answer: Claude's answer based on retrieved chunks
    """
    
    # Step 1: Chunk the document
    chunks = chunk_by_word_count(document_text, chunk_size=50, overlap=5)
    
    # Step 2: Train embeddings on the chunks
    sentences = [simple_preprocess(chunk) for chunk in chunks]
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1)
    
    # Step 3: Search for relevant chunks
    results = search_chunks(question, chunks, model, top_k=2)
    retrieved_chunks = [chunk for _, chunk, _ in results]
    context = "\n\n".join(retrieved_chunks)
    
    # Step 4: Send to Claude with retrieved context
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=500,
        system="You are a helpful assistant. Answer the user's question based ONLY on the provided documents.",
        messages=[
            {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {question}"}
        ]
    )
    
    return response.content[0].text

# Test it with different documents and questions
if __name__ == "__main__":
    # Document 1: About Python
    doc1 = """
    Python is a high-level programming language known for its simplicity and readability. 
    Created in 1991 by Guido van Rossum, it emphasizes code clarity and allows developers 
    to express concepts in fewer lines of code compared to languages like C++ or Java. 
    
    It supports multiple programming paradigms including procedural, object-oriented, and 
    functional programming. Python's extensive standard library and third-party packages 
    make it versatile for web development, data analysis, artificial intelligence, 
    scientific computing, and automation.
    """
    
    # Document 2: About AI/ML
    doc2 = """
    Machine learning is a subset of artificial intelligence where computers learn patterns 
    from data to make predictions or decisions without being explicitly programmed. 
    Libraries like TensorFlow, PyTorch, and Scikit-learn provide tools for building ML models. 
    
    Deep learning is a specialized branch of machine learning that uses neural networks 
    with multiple layers to process information. It powers applications like image recognition, 
    natural language processing, and recommendation systems.
    """
    
    # Test with both documents
    test_cases = [
        (doc1, "What is Python?"),
        (doc2, "What is machine learning?"),
        (doc1, "What can Python be used for?"),
        (doc2, "Name some deep learning libraries"),
    ]
    
    for doc, question in test_cases:
        print(f"Question: {question}")
        answer = answer_question_about_documents(doc, question)
        print(f"Answer: {answer}\n")
        print("-" * 60 + "\n")