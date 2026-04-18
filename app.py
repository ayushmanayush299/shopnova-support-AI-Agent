import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Tiny model — runs on CPU, no GPU needed
MODEL = "Qwen/Qwen2-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a friendly customer support agent for ShopNova, an online retail store. "
    "Help customers with orders, returns, shipping, and product questions. "
    "Be polite and concise. If you need an order ID, ask for it."
)

print("Loading model …")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float32,
    device_map="cpu",
)
model.eval()
print("Model ready ✅")


def respond(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for human, assistant in history:
        messages.append({"role": "user",      "content": human})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


demo = gr.ChatInterface(
    fn=respond,
    title="🛒 ShopNova Customer Support",
    description="Ask me about orders, returns, shipping, products, and more!",
    examples=[
        "What is your return policy?",
        "How do I track my order?",
        "My item arrived damaged, what do I do?",
        "Do you offer free shipping?",
        "How do I apply a promo code?",
    ],
)

if __name__ == "__main__":
    demo.launch()
