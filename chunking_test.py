# Simple chunking by word count

document = """
Python is a high-level programming language known for its simplicity and readability. 
Created in 1991 by Guido van Rossum, it emphasizes code clarity and allows developers 
to express concepts in fewer lines of code compared to languages like C++ or Java. 
Python is dynamically typed, meaning you don't need to declare variable types explicitly. 

It supports multiple programming paradigms including procedural, object-oriented, and 
functional programming. Python's extensive standard library and third-party packages 
make it versatile for web development, data analysis, artificial intelligence, 
scientific computing, and automation. Major tech companies like Google, Facebook, 
and Netflix use Python extensively.

Machine learning is a subset of artificial intelligence where computers learn patterns 
from data to make predictions or decisions without being explicitly programmed. 
Libraries like TensorFlow, PyTorch, and Scikit-learn provide tools for building ML models. 
Python has become the go-to language for machine learning because of its simplicity 
and the rich ecosystem of ML libraries. Most data scientists and ML engineers prefer Python.

Deep learning is a specialized branch of machine learning that uses neural networks 
with multiple layers to process information. It powers applications like image recognition, 
natural language processing, and recommendation systems. Training deep learning models 
requires significant computational resources, often using GPUs. PyTorch and TensorFlow 
are the dominant frameworks for deep learning today.
"""

def chunk_by_word_count(text, chunk_size=100, overlap=10):
    """Break text into chunks of roughly chunk_size words with overlap."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
    
    return chunks

# Chunk the document
chunks = chunk_by_word_count(document, chunk_size=50, overlap=5)

print(f"Total chunks: {len(chunks)}\n")
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i} ({len(chunk.split())} words):")
    print(chunk)
    print("-" * 50 + "\n")