import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2 model and tokenizer
@st.cache_resource
def load_model():
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def predict_next_word(prompt, max_new_tokens=1):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result[len(prompt):].strip()

def autocomplete(prompt, max_length=20):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=len(inputs['input_ids'][0]) + max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            num_return_sequences=1
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Streamlit UI ---
st.title("🧠 Word Prediction Tool")

prompt = st.text_input("Type your sentence:", placeholder="Start typing...")

mode = st.radio("Choose prediction mode:", ("Next Word", "Autocomplete"))

if prompt.strip():
    if mode == "Next Word":
        prediction = predict_next_word(prompt)
        st.markdown(f"**Next Word Suggestion:** `{prediction}`")
    else:
        completion = autocomplete(prompt)
        st.markdown(f"**Autocompleted Sentence:** `{completion}`")
