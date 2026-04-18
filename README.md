# 🛒 ShopNova Customer Support Chatbot

An AI-powered customer support chatbot fine-tuned on LLaMA-3 8B using Unsloth + QLoRA for e-commerce customer support.

---

## 📖 Overview

ShopNova Customer Support Chatbot is an end-to-end **LLM fine-tuning project** that simulates a real-world AI customer support system for an e-commerce store.

The project starts from scratch — generating synthetic training data, fine-tuning a large language model using parameter-efficient methods (QLoRA), and deploying it as a live chatbot accessible via a public URL.

The key idea is that instead of building a rule-based chatbot with hardcoded responses, this project trains an actual language model to **understand and respond** to customer queries naturally — just like a human support agent would.

The entire training pipeline runs on a **free Google Colab T4 GPU** in under 10 minutes, making it accessible and reproducible for anyone.

---

## 🎯 Objective

The goal of this project is to build an **AI-powered customer support chatbot** for an e-commerce store (ShopNova) that can:

- Automatically answer **frequently asked questions** (returns, shipping, payments, warranties)
- Handle **order tracking** queries in a multi-turn conversation
- Resolve **customer complaints** (damaged, wrong, missing items)
- Answer **product-specific questions** (warranty, compatibility, availability)

Instead of using a general-purpose AI, the model is **fine-tuned on custom e-commerce data** using **Unsloth + QLoRA** — making it faster, cheaper to train, and more accurate for customer support tasks compared to a generic chatbot.

The project demonstrates a **real-world LLM fine-tuning pipeline** from data generation → training → evaluation → deployment on Hugging Face Spaces.

---

## 🚀 Demo

👉 [Live Demo on Hugging Face Spaces](https://ayushmanayushdataanalyst-supportbot-llama3.hf.space)

---

## 📌 Features

- ✅ Answer customer FAQs (returns, shipping, payments, warranties)
- ✅ Handle order tracking queries in multi-turn conversations
- ✅ Resolve complaints (damaged, wrong, missing items)
- ✅ Answer product-specific questions (warranty, compatibility, availability)
- ✅ Multi-turn conversation memory
- ✅ Deployed as a public web app on Hugging Face Spaces
- ✅ Fine-tuned on 340 realistic e-commerce conversations
- ✅ Trained in under 10 minutes using QLoRA

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **LLaMA-3 8B Instruct** | Base language model |
| **Unsloth** | 2x faster fine-tuning |
| **QLoRA (PEFT)** | Memory-efficient training (4-bit) |
| **TRL (SFTTrainer)** | Supervised fine-tuning |
| **Hugging Face Hub** | Model storage |
| **Gradio** | Chatbot UI |
| **Hugging Face Spaces** | Free deployment |
| **Google Colab (T4)** | Training environment |

---

## ⚙️ How It Works

1. **Dummy data** is generated to simulate real e-commerce customer support conversations — FAQs, order tracking, complaints, and product questions
2. The data is formatted in **ShareGPT format** (system / human / gpt turns)
3. **LLaMA-3 8B** is loaded in 4-bit using Unsloth and LoRA adapters are attached to the attention and FFN layers
4. The model is fine-tuned using **SFTTrainer** for 3 epochs
5. Only the LoRA adapter weights (~150 MB) are saved — not the full 5.7 GB model
6. At inference, the base model is loaded from Hugging Face and the adapter is applied on top
7. The chatbot is served via **Gradio** on Hugging Face Spaces

---

## 🗂️ Dataset

| Type | Samples | Description |
|---|---|---|
| FAQ | 80 | Return policy, shipping, refunds, payments, account |
| Order Tracking | 120 | Multi-turn: ask for order ID → return status + ETA |
| Complaints | 60 | Damaged, wrong, missing items, double charges |
| Product Q&A | 80 | Warranty, compatibility, colours, returns |
| **Total** | **340** | 90% train / 10% eval split |

- Format: ShareGPT (system / human / gpt)
- Generated using Python with realistic dummy orders, product names, and customer names
- No real customer data used

---

## 🧠 Model Details

| Detail | Info |
|---|---|
| Base Model | LLaMA-3 8B Instruct |
| Fine-tuning Method | QLoRA (4-bit) via Unsloth |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Target Modules | q, k, v, o, gate, up, down projections |
| Training Epochs | 3 |
| Batch Size | 8 (2 × 4 accumulation) |
| Learning Rate | 2e-4 |
| Training Time | ~8 minutes on Tesla T4 |
| Peak VRAM | 11.55 GB |
| Final Eval Loss | 0.1202 |
| Adapter Size | ~150 MB |

---

## 🔄 Workflow

```
1. Generate dummy data (340 conversations)
        ↓
2. Format into ShareGPT (system/human/gpt)
        ↓
3. Load LLaMA-3 8B in 4-bit (Unsloth)
        ↓
4. Attach LoRA adapters
        ↓
5. Fine-tune with SFTTrainer (3 epochs)
        ↓
6. Save LoRA adapter (~150 MB)
        ↓
7. Upload adapter to Hugging Face Hub
        ↓
8. Deploy Gradio app on HF Spaces
        ↓
9. Public chatbot URL 🎉
```

---

## 🖥️ How to Run the Project

### Option 1 — Run on Google Colab (recommended)

```python
# 1. Install dependencies
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install transformers trl peft accelerate bitsandbytes datasets torch

# 2. Upload the script and run
!python customer_support_finetune_all_in_one.py --mode all
```

### Option 2 — Run locally (GPU required)

```bash
# 1. Clone the repo
git clone https://github.com/ayushmanayushdataanalyst/shopnova-support
cd shopnova-support

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

### Option 3 — Use the live demo

👉 [https://ayushmanayushdataanalyst-supportbot-llama3.hf.space](https://ayushmanayushdataanalyst-supportbot-llama3.hf.space)

---

## 📊 Training Results

| Epoch | Train Loss | Eval Loss |
|---|---|---|
| 1 | 0.158 | 0.136 |
| 2 | 0.135 | 0.120 |
| 3 | 0.136 | 0.120 |

Loss dropped from **2.41 → 0.12** across 3 epochs showing strong convergence.

---

## 📁 Project Structure

```
shopnova-support/
├── app.py                                    # Gradio chatbot app
├── requirements.txt                          # Dependencies
├── customer_support_finetune_all_in_one.py   # Full training pipeline
└── README.md
```

---

## 💬 Example Conversations

**Order Tracking:**
> User: Where is my order?
> Bot: I'd be happy to help! Could you provide your order ID?
> User: ORD-482910
> Bot: Your order is currently Shipped via FedEx. Expected delivery: April 20, 2026.

**FAQ:**
> User: What is your return policy?
> Bot: ShopNova offers a 30-day hassle-free return policy. Items must be unused and in original packaging...

**Complaint:**
> User: My item arrived damaged.
> Bot: I'm so sorry! Please go to My Orders → Report Issue with a photo. We'll send a replacement within 2 business days.

---

## 🔮 Future Improvements

- [ ] Connect to a **real order database** for live order tracking
- [ ] Add **RAG (Retrieval Augmented Generation)** to pull from live product catalog
- [ ] Train on **real customer support tickets** for better accuracy
- [ ] Add **multilingual support** (Hindi, Spanish, French)
- [ ] Deploy with **GPU on HF Spaces** for faster responses
- [ ] Add **voice input/output** using Whisper + TTS
- [ ] Integrate with **WhatsApp / Telegram** bot API
- [ ] Add **sentiment analysis** to detect angry customers and escalate

---

## 👤 Author

**Ayushman** — [Hugging Face](https://huggingface.co/ayushmanayushdataanalyst)
