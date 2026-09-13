from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np

document = """
Python is a high-level programming language known for its simplicity and readability. 
Created in 1991 by Guido van Rossum, it emphasizes code clarity and allows developers 
to express concepts in fewer lines of code compared to languages like C++ or Java. 

It supports multiple programming paradigms including procedural, object-oriented, and 
functional programming. Python's extensive standard library and third-party packages 
make it versatile for web development, data analysis, artificial intelligence, 
scientific computing, and automation.

Machine learning is a subset of artificial intelligence where computers learn patterns 
from data to make predictions or decisions without being explicitly programmed. 
Libraries like TensorFlow, PyTorch, and Scikit-learn provide tools for building ML models. 

Deep learning is a specialized branch of machine learning that uses neural networks 
with multiple layers to process information. It powers applications like image recognition, 
natural language processing, and recommendation systems.
"""

def chunk_by_word_count(text, chunk_size=100, overlap=10):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
    return chunks

def get_chunk_embedding(chunk, model):
    """Average word vectors to get chunk embedding."""
    words = simple_preprocess(chunk)
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)

def search_chunks(question, chunks, model, top_k=2):
    """Find the most relevant chunks for a question."""
    question_embedding = get_chunk_embedding(question, model)
    
    similarities = []
    for i, chunk in enumerate(chunks):
        chunk_embedding = get_chunk_embedding(chunk, model)
        sim = cosine_similarity(question_embedding, chunk_embedding)
        similarities.append((i, chunk, sim))
    
    # Sort by similarity, highest first
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    # Return top_k results
    return similarities[:top_k]

# Set up
chunks = chunk_by_word_count(document, chunk_size=50, overlap=5)
sentences = [simple_preprocess(chunk) for chunk in chunks]
model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1)

# Test searches
questions = [
    "What is machine learning?",
    "Tell me about Python",
    "How does deep learning work?"
]

for question in questions:
    print(f"Question: {question}")
    results = search_chunks(question, chunks, model, top_k=2)
    for rank, (chunk_idx, chunk_text, similarity) in enumerate(results, 1):
        print(f"  [{rank}] Chunk {chunk_idx+1} (similarity: {similarity:.3f})")
        print(f"      {chunk_text[:80]}...\n")
    print("-" * 60 + "\n")