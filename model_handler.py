import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Tuple

@st.cache_resource
def load_distilgpt2_model():
    try:
        # Try to load real model first
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
        
        return {"type": "real", "model": generator}
    except Exception as e:
        st.warning(f"Could not load real model: {str(e)}")
        st.info("Using mock model for demonstration purposes")
        return {"type": "mock", "model": None}

def generate_response(query: str, model_wrapper) -> Tuple[str, float]:
    if model_wrapper is None:
        return "Model not loaded properly", 0.0
    
    try:
        if model_wrapper["type"] == "real":
            # Use real model
            generator = model_wrapper["model"]
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
            
        else:
            # Use mock model
            generated_text = generate_mock_response(query)
            confidence_score = 0.75
        
        return generated_text, confidence_score
        
    except Exception as e:
        return f"Error generating response: {str(e)}", 0.0

def generate_mock_response(query: str) -> str:
    """Generate a mock response for demonstration purposes"""
    import random
    
    # Simple keyword-based responses
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['stock', 'investment', 'market', 'finance']):
        responses = [
            "Based on current market conditions, investments require careful analysis of risk factors and potential returns.",
            "Stock market performance depends on various economic indicators and company fundamentals.",
            "Financial markets are influenced by multiple factors including economic policies and global events."
        ]
    elif any(word in query_lower for word in ['python', 'programming', 'code']):
        responses = [
            "Python is a versatile programming language widely used in data science and machine learning applications.",
            "Programming involves writing instructions for computers to execute specific tasks efficiently.",
            "Code quality depends on factors like readability, maintainability, and performance optimization."
        ]
    elif any(word in query_lower for word in ['machine learning', 'ai', 'artificial intelligence']):
        responses = [
            "Machine learning algorithms learn patterns from data to make predictions or decisions.",
            "AI systems can process large amounts of information and identify complex relationships.",
            "Artificial intelligence applications span across various industries from healthcare to finance."
        ]
    else:
        responses = [
            f"Based on your query about '{query[:50]}...', this topic involves multiple considerations and factors.",
            f"The question regarding '{query[:30]}...' requires analysis of various related aspects.",
            f"Your inquiry about '{query[:40]}...' touches on important concepts that merit detailed examination."
        ]
    
    return random.choice(responses)

def get_model_info():
    return {
        "model_name": "DistilGPT-2",
        "description": "Distilled version of GPT-2, smaller and faster",
        "parameters": "82M",
        "size": "~350MB"
    }