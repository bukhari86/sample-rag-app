from dotenv import load_dotenv
import os
from anthropic import Anthropic
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key)

# RAG functions from day 3
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
        system="You are a helpful assistant. Answer the user's question based ONLY on the provided documents. If information is not in the documents, say 'not mentioned'.",
        messages=[
            {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {question}"}
        ]
    )
    
    return response.content[0].text

# Our test documents
documents = {
    "python_and_ml": """
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
    and the rich ecosystem of ML libraries.

    Deep learning is a specialized branch of machine learning that uses neural networks 
    with multiple layers to process information. It powers applications like image recognition, 
    natural language processing, and recommendation systems. Training deep learning models 
    requires significant computational resources, often using GPUs.
    """
}

# Test cases
test_cases = [
    ("What is Python?", ["high-level", "programming language", "simplicity", "readability"]),
    ("When was Python created?", ["1991", "Guido van Rossum"]),
    ("What programming paradigms does Python support?", ["procedural", "object-oriented", "functional"]),
    ("What can Python be used for?", ["web development", "data analysis", "artificial intelligence"]),
    ("Who created Python?", ["Guido van Rossum"]),
    ("What is machine learning?", ["artificial intelligence", "learn patterns", "data", "decisions"]),
    ("Name some machine learning libraries", ["TensorFlow", "PyTorch", "Scikit-learn"]),
    ("What is the difference between machine learning and deep learning?", ["neural networks", "multiple layers", "subset"]),
    ("What are neural networks used for?", ["image recognition", "natural language processing", "recommendation"]),
    ("What is deep learning?", ["neural networks", "multiple layers", "machine learning"]),
    ("List all Python libraries mentioned", ["standard library", "third-party packages"]),
    ("What companies use Python?", ["Google", "Facebook", "Netflix"]),
    ("How does Python differ from C++ and Java?", ["fewer lines of code", "simplicity"]),
    ("What are the benefits of deep learning?", ["image recognition", "natural language processing"]),
    ("Why is Python popular for machine learning?", ["simplicity", "ecosystem", "libraries"]),
    ("What is reinforcement learning?", ["not mentioned"]),
    ("How much does Python cost?", ["not mentioned"]),
    ("What is the latest Python version?", ["not mentioned"]),
    ("Which is better, Python or JavaScript?", ["not mentioned"]),
    ("What is a neural network?", ["neurons", "layers", "machine learning", "deep learning"]),
]

# Run all tests
print("Running 20 test cases...\n")
print("=" * 80 + "\n")

results = []
for i, (question, expected_keywords) in enumerate(test_cases, 1):
    print(f"Test {i}: {question}")
    answer = answer_question_about_documents(documents["python_and_ml"], question)
    print(f"Answer: {answer}\n")
    results.append((question, answer, expected_keywords))
    print("-" * 80 + "\n")

# Save results for scoring
import json
with open("test_results.json", "w") as f:
    json.dump([{"question": q, "answer": a, "expected": e} for q, a, e in results], f, indent=2)

print(f"\nResults saved to test_results.json")