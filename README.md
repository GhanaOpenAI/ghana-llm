<div align="center">

# 🇬🇭 Ghana-LLM

**A Foundational Bilingual Language Model for English & Asante Twi (Akan)**

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-ghana--llm--base-blue)](https://huggingface.co/ghanaopenai/ghana-llm-base)
[![Hugging Face Collection](https://img.shields.io/badge/🤗%20Collection-MiniMind%20English--Twi-orange)](https://huggingface.co/collections/ghanaopenai/minimind-english-twi-6adb2ea628e5ba331db321ad)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

*Developed by [Ghana Open AI](https://huggingface.co/ghanaopenai) & supported by [Ghana NLP](https://ghananlp.org)*

</div>

---

## 📌 Overview

**Ghana-LLM** is an open-source, lightweight bilingual language model built specifically for Ghanaian language computing. Trained from scratch on over **4.85 million passages (~7.9 GB / ~1.5 billion tokens)** of balanced Asante Twi and English text, Ghana-LLM delivers high-quality natural language understanding and generation for low-resource environments.

### Key Capabilities
- **Bilingual Input ➔ Twi Output**: Understands user inputs in both **English** and **Asante Twi**, while generating fluent, high-quality responses exclusively in Asante Twi.
- **Context-Grounded Retrieval & RAG**: Specifically conditioned to ingest multi-source search results (such as from [Ghana Search](https://github.com/GhanaOpenAI/ghana-search)) in English and synthesize accurate, grounded answers in Asante Twi with anti-hallucination guardrails.
- **Compact & Edge-Ready**: At **64M parameters**, the model requires under **1 GB of VRAM**, enabling real-time inference on smartphones, Raspberry Pis, low-cost servers, and browser environments.

---

## 🏗️ Model Architecture

| Specification | Configuration |
|---|---|
| **Architecture** | Transformer Decoder-Only (Pre-Norm RMSNorm, SwiGLU, RoPE) |
| **Parameters** | 63.91 Million |
| **Hidden Dimension ($d_{\text{model}}$)** | 768 |
| **Layers** | 8 |
| **Attention Heads** | 8 Query heads, 4 Key-Value heads (Grouped-Query Attention) |
| **Context Window** | 32,768 tokens (trained at 512, YaRN extrapolation supported) |
| **Vocabulary Size** | 6,400 (Byte-Level BPE with native Akan character mapping) |
| **Base Weight** | [`ghanaopenai/ghana-llm-base`](https://huggingface.co/ghanaopenai/ghana-llm-base) |

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/GhanaOpenAI/ghana-llm.git
cd ghana-llm
pip install -r requirements.txt
```

### 2. Interactive CLI Chat

Start an interactive streaming chat session directly with the model:

```bash
python scripts/inference.py --model_name_or_path ghanaopenai/ghana-llm-base
```

You can prompt the model in English or Asante Twi:
```text
User 💬: How does a solar panel generate electricity?
Ghana-LLM 🇬🇭: Solar panel yɛ anyinam ahoɔden mfiri a ɛde owia hann dan anyinam ahoɔden. Sɛ owia bɔ so a...

User 💬: Ɛhe na me nsa bɛka Ghana Immigration Service ti dwumadibea?
Ghana-LLM 🇬🇭: Ghana Immigration Service ti dwumadibea no wɔ Off Starlets 91 Road, Ministries, Accra...
```

### 3. Python API Integration

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "ghanaopenai/ghana-llm-base"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, 
    trust_remote_code=True
).to(device)

messages = [{"role": "user", "content": "Kyerɛ me sɛnea wosi dua aborɔdeɛ yiye."}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(
    **inputs, 
    max_new_tokens=256, 
    temperature=0.7, 
    top_p=0.9, 
    do_sample=True,
    repetition_penalty=1.1
)

print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

### 4. OpenAI-Compatible API Server

Launch a high-performance local server compatible with the OpenAI API protocol:

```bash
python scripts/serve_api.py --port 8000
```

Connect using standard client libraries:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

response = client.chat.completions.create(
    model="ghana-llm",
    messages=[{"role": "user", "content": "What is the capital of Ghana?"}]
)
print(response.choices[0].message.content)
```

---

## 🛠️ Fine-Tuning Guide

Ghana-LLM provides complete scripts for parameter-efficient LoRA fine-tuning, full-parameter SFT, and alignment.

### 1. Data Format

Data should follow the unified multi-turn conversation format:

```json
{"conversations": [{"role": "user", "content": "How do I renew my driver's license?"}, {"role": "assistant", "content": "Sɛ wopɛ sɛ woyɛ wo lセンス foforo a, kɔ DVLA dwumadibea a ɛbɛn wo..."}]}
```

### 2. Parameter-Efficient Fine-Tuning (LoRA)

Fine-tune on consumer GPUs (RTX 3060/3090/4090 or Apple Silicon):

```bash
cd trainer
python train_lora.py \
    --from_weight none \
    --data_path ../examples/custom_dataset_sample.jsonl \
    --lora_name lora_my_domain \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --max_seq_len 768 \
    --dtype bfloat16
```

### 3. Full-Parameter SFT Fine-Tuning

```bash
cd trainer
python train_full_sft.py \
    --from_weight pretrain \
    --data_path ../dataset/sft_t2t.jsonl \
    --epochs 2 \
    --batch_size 32 \
    --accumulation_steps 4 \
    --learning_rate 2e-5 \
    --max_seq_len 768 \
    --dtype bfloat16
```

### 4. Cross-Lingual DPO Alignment

Enforce strict output policy (rewarding Asante Twi, penalizing English output):

```bash
cd trainer
python train_dpo.py \
    --from_weight full_sft \
    --data_path ../dataset/dpo.jsonl \
    --epochs 2 \
    --batch_size 8 \
    --learning_rate 4e-8 \
    --beta 0.15
```

---

## 🔍 Context-Grounded Search / RAG Integration

Ghana-LLM is optimized for retrieval engines like [Ghana Search](https://github.com/GhanaOpenAI/ghana-search):

```python
search_results = """
[1] Title: Ghana Revenue Authority (GRA)
Category: Public Services | Address: Ministries, Accra
Digital Address: GA-144-1234 | Website: https://gra.gov.gh
Snippet: Individual income tax filing deadline is April 30 annually.
"""

user_query = "When is the deadline to file individual income tax in Ghana?"

prompt = f"""Context:
{search_results}

Question: {user_query}
Answer the question in Asante Twi based strictly on the provided context."""

# Pass prompt to Ghana-LLM -> outputs accurate grounded response in Asante Twi
```

---

## 📦 Datasets

All datasets used to build Ghana-LLM are open-sourced under the [**MiniMind English-Twi Collection**](https://huggingface.co/collections/ghanaopenai/minimind-english-twi-6adb2ea628e5ba331db321ad):

* [`ghanaopenai/minimind-pretrain-en-tw`](https://huggingface.co/datasets/ghanaopenai/minimind-pretrain-en-tw): 4.85M passages (7.9 GB)
* [`ghanaopenai/minimind-sft-en-tw`](https://huggingface.co/datasets/ghanaopenai/minimind-sft-en-tw): 1.89M dialogues (4.6 GB)
* [`ghanaopenai/minimind-rag-sft-en-tw`](https://huggingface.co/datasets/ghanaopenai/minimind-rag-sft-en-tw): 378k RAG dialogues (968 MB)
* [`ghanaopenai/minimind-dpo-en-tw`](https://huggingface.co/datasets/ghanaopenai/minimind-dpo-en-tw): 36k preference pairs (65 MB)
* [`ghanaopenai/minimind-agent-math-en-tw`](https://huggingface.co/datasets/ghanaopenai/minimind-agent-math-en-tw): 40k verifiable items (37 MB)

---

## 📄 License & Attribution

This project is licensed under the [Apache License 2.0](LICENSE). 
Pretraining and fine-tuning utilize community datasets published by [Ghana NLP](https://ghananlp.org) and [Ghana Open AI](https://huggingface.co/ghanaopenai).
