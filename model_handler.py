import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Tuple

@st.cache_resource
def load_distilgpt2_model():
    try:
        model_name = "distilgpt2"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            return_full_text=False,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=150,
            pad_token_id=tokenizer.eos_token_id
        )
        
        return generator
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def generate_response(query: str, generator) -> Tuple[str, float]:
    if generator is None:
        return "Model not loaded properly", 0.0
    
    try:
        prompt = f"Question: {query}\nAnswer:"
        
        result = generator(prompt, max_new_tokens=100, num_return_sequences=1)
        
        generated_text = result[0]['generated_text'].strip()
        
        if generated_text.startswith("Question:") or generated_text.startswith("Answer:"):
            lines = generated_text.split('\n')
            for line in lines:
                if line.strip() and not line.startswith("Question:") and not line.startswith("Answer:"):
                    generated_text = line.strip()
                    break
        
        confidence_score = min(0.95, max(0.5, len(generated_text) / 200))
        
        return generated_text, confidence_score
        
    except Exception as e:
        return f"Error generating response: {str(e)}", 0.0

def get_model_info():
    return {
        "model_name": "DistilGPT-2",
        "description": "Distilled version of GPT-2, smaller and faster",
        "parameters": "82M",
        "size": "~350MB"
    }