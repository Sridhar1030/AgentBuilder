import os
import json
import time
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
# 1. SETUP
GROQ_API_KEY = os.getenv("API_KEY")
client = Groq(api_key=GROQ_API_KEY)
MODEL = "llama-3.3-70b-versatile"  # The "Teacher"

# 2. YOUR TOPICS (Expand this to 50 items)
topics = [
    "Write Python code to print 'hello world' with each letter on a different line.",
    "Write a Python function that takes a list of numbers and returns the sum.",
    "Write Python code to print the first 10 Fibonacci numbers.",
    "Write a Python function that reverses a string without using [::-1] or reversed().",
    "Write Python code to check if a number is prime and print True or False.",
    "Write a Python list comprehension that squares every element in a list of integers.",
    "Write a Python function that counts how many times each character appears in a string.",
    "Write Python code to read a file and print its contents line by line.",
    "Write a Python function that flattens a nested list (one level deep).",
    "Write Python code to print a multiplication table for numbers 1 through 5.",
    "Write a Python function that finds the maximum value in a list without using max().",
    "Write Python code to print all even numbers from 0 to 20 using a for loop.",
    "Write a Python function that takes two strings and returns True if one is an anagram of the other.",
    "Write Python code to capitalize the first letter of each word in a string.",
    "Write a Python function that removes duplicates from a list while preserving order.",
    "Write Python code to print a right-angled triangle of asterisks with 5 rows.",
    "Write a Python function that merges two sorted lists into one sorted list.",
    "Write Python code to convert a list of integers to a single comma-separated string.",
    "Write a Python function that returns the factorial of a non-negative integer.",
    "Write Python code to print the current date and time.",
    "Write a Python function that checks if a string is a palindrome.",
    "Write Python code to add all key-value pairs from dict2 into dict1 and print the result.",
    "Write a Python function that returns the second largest number in a list.",
    "Write Python code to print every third character of a string starting from index 0.",
    "Write a Python function that takes a list and returns only the elements at even indices.",
    "Write Python code to ask the user for their name and print 'Hello, <name>'.",
    "Write a Python function that counts vowels (a, e, i, o, u) in a string.",
    "Write Python code to create a dict mapping each word in a string to its length.",
    "Write a Python function that returns True if a list is sorted in ascending order.",
    "Write Python code to print numbers 1 to 10 using a while loop.",
    "Write a Python function that finds the longest word in a list of strings.",
    "Write Python code to swap the first and last elements of a list in place.",
    "Write a Python function that returns the intersection of two lists (common elements).",
    "Write Python code to print a list of numbers formatted with 2 decimal places.",
    "Write a Python function that converts a list of tuples into a dictionary.",
    "Write Python code to print the keys of a dictionary sorted alphabetically.",
    "Write a Python function that returns the nth element of the Fibonacci sequence.",
    "Write Python code to split a string on spaces and print each word on a new line.",
    "Write a Python function that rotates a list left by n positions (e.g. [1,2,3,4], 2 -> [3,4,1,2]).",
    "Write Python code to print a 3x3 grid of numbers 1 through 9 using nested loops.",
    "Write a Python function that returns the median of a list of numbers.",
    "Write Python code to replace all occurrences of 'cat' with 'dog' in a string.",
    "Write a Python function that returns the first N elements of a list, or the whole list if len(list) < N.",
    "Write Python code to create a set from a list and print its size.",
    "Write a Python function that returns the indices of all occurrences of a value in a list.",
    "Write Python code to concatenate two lists and sort the result.",
    "Write a Python function that returns True if two lists have at least one common element.",
    "Write Python code to print the ASCII value of each character in 'Python'.",
    "Write a Python function that pads a string with spaces to a given length (left-justified).",
    "Write Python code to compute and print the sum of digits of an integer.",
    "Write a Python function that returns the union of two lists (all unique elements).",
    "Write Python code to print a countdown from 10 to 1 then print 'Go!'.",
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
