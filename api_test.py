from dotenv import load_dotenv
import os
from anthropic import Anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key)

def summarize_file(filename):
    """Read a file and return a Claude-generated summary."""
    with open(filename) as file:
        document_text = file.read()
    
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=500,
        system="You are a document summarizer. Provide a concise summary in 2-3 sentences.",
        messages=[
            {"role": "user", "content": f"Summarize this document:\n\n{document_text}"}
        ]
    )
    
    return response.content[0].text

# Test 1: Summarize sample.txt
print("Test 1: sample.txt")
summary1 = summarize_file("sample.txt")
print(summary1)
print("\n" + "="*50 + "\n")

# Test 2: Create a second file and summarize it
# First, create the file
with open("sample2.txt", "w") as f:
    f.write("Artificial Intelligence is transforming industries by automating complex tasks, improving decision-making, and enabling new capabilities. Machine learning, a subset of AI, allows systems to learn from data without explicit programming. Deep learning uses neural networks with multiple layers to process information in sophisticated ways. AI applications range from medical diagnostics to autonomous vehicles to natural language processing.")

print("Test 2: sample2.txt")
summary2 = summarize_file("sample2.txt")
print(summary2)