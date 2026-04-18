# 🛒 ShopNova Customer Support Chatbot

An AI-powered customer support chatbot fine-tuned on LLaMA-3 8B using Unsloth + QLoRA for e-commerce customer support.

---

## 🚀 Demo

👉 [Live Demo on Hugging Face Spaces](https://ayushmanayushdataanalyst-supportbot-llama3.hf.space)

---

## 📌 Features

- Answer customer FAQs (returns, shipping, payments, warranties)
- Handle order tracking queries
- Resolve complaints (damaged, wrong, missing items)
- Answer product-specific questions
- Multi-turn conversation support

---

## 🧠 Model Details

| Detail | Info |
|---|---|
| Base Model | LLaMA-3 8B Instruct |
| Fine-tuning Method | QLoRA (4-bit) via Unsloth |
| Training Data | 340 dummy e-commerce conversations |
| Training Time | ~8 minutes on Tesla T4 |
| Peak VRAM | 11.55 GB |
| Final Eval Loss | 0.1202 |

---

## 📁 Project Structure

```
shopnova-support/
├── app.py               # Gradio chatbot app
├── requirements.txt     # Dependencies
└── README.md
```

---

## 🛠️ Tech Stack

- **Unsloth** — 2x faster fine-tuning
- **LLaMA-3 8B** — base model
- **QLoRA (PEFT)** — memory-efficient training
- **Gradio** — chatbot UI
- **Hugging Face Spaces** — deployment

---

## 📊 Training Results

| Epoch | Train Loss | Eval Loss |
|---|---|---|
| 1 | 0.158 | 0.136 |
| 2 | 0.135 | 0.120 |
| 3 | 0.136 | 0.120 |

---

## 💬 Example Conversations

**Order Tracking:**
> User: Where is my order?
> Bot: I'd be happy to help! Could you provide your order ID?
> User: ORD-482910
> Bot: Your order is currently Shipped via FedEx. Expected delivery: April 20, 2026.

**FAQ:**
> User: What is your return policy?
> Bot: ShopNova offers a 30-day hassle-free return policy...

---

## 👤 Author

**Ayushman** — [Hugging Face](https://huggingface.co/ayushmanayushdataanalyst)
