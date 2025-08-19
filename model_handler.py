import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple

@st.cache_resource
def load_gpt2_small_model():
    try:
        model_name = "gpt2"
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Return model and tokenizer separately for better control
        return {"model": model, "tokenizer": tokenizer}
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def generate_response(query: str, model_dict) -> Tuple[str, float]:
    if model_dict is None:
        return "Model not loaded properly", 0.0
    
    try:
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        
        prompt = f"You are a helpful assistant. Answer the following question clearly and concisely.\n\nQuestion: {query}\nAnswer:"
        
        # Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        
        # Generate with output scores to get probabilities
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode generated text
        generated_tokens = outputs.sequences[0][len(inputs[0]):]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Calculate real confidence from token probabilities
        confidence_score = calculate_confidence_from_scores(outputs.scores)
        
        # Clean up the response
        if "Question:" in generated_text or "Answer:" in generated_text:
            lines = generated_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith("Question:") and not line.startswith("Answer:"):
                    generated_text = line
                    break
        
        # Post-process to remove repetitions
        generated_text = remove_repetitions(generated_text)
        
        # Truncate at natural sentence ending
        generated_text = truncate_at_sentence_end(generated_text)
        
        return generated_text, confidence_score
        
    except Exception as e:
        return f"Error generating response: {str(e)}", 0.0

def calculate_confidence_from_scores(scores) -> float:
    """Calculate confidence score from model output probabilities"""
    if not scores:
        return 0.5
    
    try:
        # Convert logits to probabilities for each token
        all_probs = []
        for score in scores:
            probs = F.softmax(score, dim=-1)
            max_prob = torch.max(probs).item()
            all_probs.append(max_prob)
        
        # Calculate average confidence
        avg_confidence = np.mean(all_probs)
        
        # Scale to reasonable range (0.3 to 0.9)
        scaled_confidence = 0.3 + (avg_confidence * 0.6)
        
        return round(float(scaled_confidence), 3)
        
    except Exception:
        return 0.5

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