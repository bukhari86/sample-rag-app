from dotenv import load_dotenv
import os
from anthropic import Anthropic

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key)

# Embed a simple sentence
text = "The cat sat on the mat"

response = client.messages.create(
    model="claude-sonnet-5",
    input=text
)

embedding = response.embedding
print(f"Embedding of '{text}':")
print(f"Length: {len(embedding)}")
print(f"First 10 numbers: {embedding[:10]}")