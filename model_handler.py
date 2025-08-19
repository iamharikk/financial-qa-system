import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Tuple

@st.cache_resource
def load_gpt2_small_model():
    try:
        model_name = "gpt2"
        
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
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        return generator
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def generate_response(query: str, generator) -> Tuple[str, float]:
    if generator is None:
        return "Model not loaded properly", 0.0
    
    try:
        prompt = f"You are a helpful assistant. Answer the following question clearly and concisely.\n\nQuestion: {query}\nAnswer:"
        
        result = generator(
            prompt, 
            max_new_tokens=80, 
            num_return_sequences=1,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3
        )
        
        generated_text = result[0]['generated_text'].strip()
        
        # Clean up the response
        if generated_text.startswith("Question:") or generated_text.startswith("Answer:"):
            lines = generated_text.split('\n')
            for line in lines:
                if line.strip() and not line.startswith("Question:") and not line.startswith("Answer:"):
                    generated_text = line.strip()
                    break
        
        # Post-process to remove repetitions
        generated_text = remove_repetitions(generated_text)
        
        # Truncate at natural sentence ending
        generated_text = truncate_at_sentence_end(generated_text)
        
        confidence_score = min(0.95, max(0.5, len(generated_text) / 150))
        
        return generated_text, confidence_score
        
    except Exception as e:
        return f"Error generating response: {str(e)}", 0.0

def remove_repetitions(text: str) -> str:
    """Remove repeated phrases or sentences"""
    sentences = text.split('. ')
    unique_sentences = []
    seen_sentences = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence not in seen_sentences:
            unique_sentences.append(sentence)
            seen_sentences.add(sentence)
        elif sentence in seen_sentences:
            break  # Stop at first repetition
    
    return '. '.join(unique_sentences)

def truncate_at_sentence_end(text: str) -> str:
    """Truncate text at the last complete sentence"""
    if not text:
        return text
    
    # Find the last occurrence of sentence-ending punctuation
    last_period = text.rfind('.')
    last_exclamation = text.rfind('!')
    last_question = text.rfind('?')
    
    last_punct = max(last_period, last_exclamation, last_question)
    
    if last_punct > 0 and last_punct < len(text) - 1:
        return text[:last_punct + 1].strip()
    
    return text.strip()

def get_model_info():
    return {
        "model_name": "GPT-2 Small",
        "description": "Small version of GPT-2 with better Q&A capabilities",
        "parameters": "124M",
        "size": "~500MB"
    }