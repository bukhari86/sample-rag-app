# Test cases for our RAG system
# Format: (question, expected_answer_keywords)
# Keywords are what should appear in a correct answer

test_cases = [
    # Python questions
    (
        "What is Python?",
        ["high-level", "programming language", "simplicity", "readability"]
    ),
    (
        "When was Python created?",
        ["1991", "Guido van Rossum"]
    ),
    (
        "What programming paradigms does Python support?",
        ["procedural", "object-oriented", "functional"]
    ),
    (
        "What can Python be used for?",
        ["web development", "data analysis", "artificial intelligence"]
    ),
    (
        "Who created Python?",
        ["Guido van Rossum"]
    ),
    
    # Machine Learning questions
    (
        "What is machine learning?",
        ["artificial intelligence", "learn patterns", "data", "decisions"]
    ),
    (
        "Name some machine learning libraries",
        ["TensorFlow", "PyTorch", "Scikit-learn"]
    ),
    (
        "What is the difference between machine learning and deep learning?",
        ["neural networks", "multiple layers", "subset"]
    ),
    (
        "What are neural networks used for?",
        ["image recognition", "natural language processing", "recommendation"]
    ),
    (
        "What is deep learning?",
        ["neural networks", "multiple layers", "machine learning"]
    ),
    
    # Harder questions (to test retrieval limits)
    (
        "List all Python libraries mentioned",
        ["standard library", "third-party packages"]
    ),
    (
        "What companies use Python?",
        ["Google", "Facebook", "Netflix"]
    ),
    (
        "How does Python differ from C++ and Java?",
        ["fewer lines of code", "simplicity"]
    ),
    (
        "What are the benefits of deep learning?",
        ["image recognition", "natural language processing"]
    ),
    (
        "Why is Python popular for machine learning?",
        ["simplicity", "ecosystem", "libraries"]
    ),
    
    # Edge cases (might fail)
    (
        "What is reinforcement learning?",
        ["not mentioned"]  # This won't be in the documents
    ),
    (
        "How much does Python cost?",
        ["not mentioned"]  # This won't be in the documents
    ),
    (
        "What is the latest Python version?",
        ["not mentioned"]  # This won't be in the documents
    ),
    (
        "Which is better, Python or JavaScript?",
        ["not mentioned"]  # This won't be in the documents
    ),
    (
        "What is a neural network?",
        ["neurons", "layers", "machine learning", "deep learning"]
    ),
]

print(f"Total test cases: {len(test_cases)}")
for i, (question, keywords) in enumerate(test_cases, 1):
    print(f"{i}. Q: {question}")
    print(f"   Expected keywords: {keywords}\n")