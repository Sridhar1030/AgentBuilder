# 🚀 1B LLM Distillation Project

An end-to-end pipeline demonstrating how to distill knowledge from a massive 70B parameter Teacher model into a lightweight 1B parameter Student model that runs entirely offline.

## 🧠 The Tech Stack
* **The Teacher:** Llama-3-70B (Served via Groq API for ultra-fast data generation)
* **The Training:** Hugging Face `trl` / `SFTTrainer` (QLoRA fine-tuning on a Google Colab T4 GPU)
* **The Student:** unsloth/Llama-3.2-1B-Instruct
* **The UI:** Gradio + Hugging Face `pipeline`

## 📂 Repository Structure
* `datagen.py` - The script that prompts the Groq API to generate synthetic training data.
* `train_data.json` - The generated instruction/response pairs.
* `app.py` - The local Gradio UI that loads the fine-tuned PEFT adapters onto the base model.
* `student_model/` - (Not uploaded due to size) The trained LoRA weights.