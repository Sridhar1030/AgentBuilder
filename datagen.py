import os
import json
import time
from groq import Groq

# 1. SETUP
GROQ_API_KEY = os.getenv("API_KEY")
client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.3-70b-versatile"  # The "Teacher"

# 2. YOUR TOPICS (Expand this to 50 items)
topics = [
    "Explain how a mortgage works to a 10 year old.",
    "Write a Python function to find the shortest path in a graph.",
    "Summarize the plot of Inception in 3 sentences.",
    "Give me a recipe for high-protein vegan pancakes.",
    "Explain photosynthesis like a pirate.",
    "How does a transformer work?",
    "What is quantum entanglement in simple terms?",
    "Explain recursion to a five-year-old.",
    "How does the internet actually work?",
    "What causes a rainbow to form?",
    "Explain gradient descent like I'm a chef.",
    "How does a CPU execute instructions?",
    "What is the difference between a virus and a bacterium?",
    "Explain blockchain without using the word 'block' or 'chain'.",
    "How do airplanes stay in the air?",
    "What happens inside a black hole?",
    "Explain DNA replication like a sports commentator.",
    "How does encryption keep my data safe?",
    "What is the theory of relativity in plain English?",
    "How do vaccines train the immune system?",
    "Explain the water cycle like a bedtime story.",
    "What is a neural network and how does it learn?",
    "How does a refrigerator cool things down?",
    "What causes earthquakes?",
    "Explain object-oriented programming using a pizza shop analogy.",
    "How does GPS know where I am?",
    "What is CRISPR and why does it matter?",
    "Explain the stock market to a teenager.",
    "How do noise-canceling headphones work?",
    "What is dark matter and why can't we see it?",
    "Explain how a compiler works in simple terms.",
    "How does the human eye perceive color?",
    "What is the difference between machine learning and deep learning?",
    "How do batteries store and release energy?",
    "Explain the greenhouse effect like a detective story.",
    "What is an API and why should I care?",
    "How does 3D printing actually work?",
    "What causes the tides in the ocean?",
    "Explain containerization in software like a shipping analogy.",
    "How does the brain form and retrieve memories?",
    "What is nuclear fusion and why is it so hard?",
    "Explain the Pythagorean theorem like a rapper.",
    "How do touch screens detect your finger?",
    "What is the microbiome and why is it important?",
    "Explain version control like a time travel movie.",
    "How does Wi-Fi transmit data through the air?",
    "What is inflation and what causes it?",
    "Explain the halting problem to a non-programmer.",
    "How does solar energy get converted into electricity?",
    "Explain the CAP theorem using a real-world analogy.",
    "How do self-driving cars perceive their environment?",
    "What is the difference between TCP and UDP?",
    "Explain evolution by natural selection like a cooking show.",
]


def generate_sample(topic):
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful and concise assistant."},
                {"role": "user", "content": topic}
            ],
            temperature=0.7,
            max_tokens=512,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error on topic '{topic}': {e}")
        return None


# 3. EXECUTION LOOP
dataset = []
print(f"Generating {len(topics)} samples...")

for i, topic in enumerate(topics):
    print(f"[{i+1}/{len(topics)}] Processing: {topic[:30]}...")
    response = generate_sample(topic)

    if response:
        # Format for SFTTrainer (Instruction-Output style)
        sample = {
            "instruction": topic,
            "output": response,
            "text": f"### Instruction:\n{topic}\n\n### Response:\n{response}"
        }
        dataset.append(sample)

    # Respect rate limits - Groq is fast, but 50 requests in 1 sec might trigger a limit
    time.sleep(0.5)

# 4. SAVE TO FILE
file_name = "train_data.json"
with open(file_name, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4)

print(f"Done! Dataset saved to {file_name}")
