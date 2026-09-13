from dotenv import load_dotenv
import os
from anthropic import Anthropic
import json

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key)

# Step 1: Define the tool (the function Claude can call)
def calculate(operation, a, b):
    """Simple calculator: add, subtract, multiply, divide."""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            return "Error: division by zero"
        return a / b
    else:
        return "Unknown operation"

# Step 2: Describe the tool to Claude (in API format)
tools = [
    {
        "name": "calculator",
        "description": "Performs basic arithmetic operations",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "The operation: add, subtract, multiply, or divide"
                },
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number"
                }
            },
            "required": ["operation", "a", "b"]
        }
    }
]

# Step 3: Ask Claude a question that requires the calculator
question = "What is 15% of 200?"

print(f"Question: {question}\n")

# Send the question with tools available
response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=1000,
    tools=tools,
    messages=[
        {"role": "user", "content": question}
    ]
)

# Step 4: Check if Claude wants to use a tool
print("Claude's response:")
for block in response.content:
    if hasattr(block, 'text'):
        print(f"Text: {block.text}")
    elif block.type == "tool_use":
        print(f"Tool call: {block.name}")
        print(f"  Input: {block.input}")
        
        # Step 5: Call the actual function
        result = calculate(**block.input)
        print(f"  Result: {result}")