import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

print("Loading Base Model... (This takes a few seconds)")
# 1. Load the Base Model (Unquantized so it works on any laptop CPU/GPU)
base_model_id = "unsloth/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto", # Will auto-detect if you have Mac M-chip, PC GPU, or just CPU
    torch_dtype=torch.float32 # Safe for all laptops
)

print("Attaching your custom Student Knowledge...")
# 2. Attach your fine-tuned adapters
model = PeftModel.from_pretrained(base_model, "./student_model")

# 3. Create the text generator
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.7
)

# 4. The Gradio Chat Logic
def chat(message, history):
    # Match the exact prompt format you trained on
    prompt = f"### Instruction:\n{message}\n\n### Response:\n"
    
    # Generate the response
    result = pipe(prompt)[0]['generated_text']
    
    # Clean up the output to only show the assistant's reply
    reply = result.split("### Response:\n")[-1]
    return reply

# 5. Launch the UI
print("Launching UI...")
demo = gr.ChatInterface(
    fn=chat,
    title="🚀 My 1B Distilled Student",
    description="Trained in 24 hours via Llama-3-70B Distillation.",
    theme="glass" # Makes it look very modern for a demo
)

if __name__ == "__main__":
    demo.launch()