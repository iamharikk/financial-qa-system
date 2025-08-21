import torch
import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

class FineTunedModelHandler:
    """Handler for the fine-tuned GPT-2 model"""
    
    def __init__(self, model_path="./fine_tuned_sft_model"):
        """Initialize the fine-tuned model handler"""
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            print(f"Loading fine-tuned model from: {self.model_path}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Ensure padding token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model_loaded = True
            print("Fine-tuned model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading fine-tuned model: {str(e)}")
            self.model_loaded = False
            return False
    
    def generate_response(self, question, max_length=100, temperature=0.3):
        """
        Generate response using the fine-tuned model
        
        Args:
            question (str): The input question
            max_length (int): Maximum length for generation
            temperature (float): Temperature for sampling
            
        Returns:
            tuple: (answer, confidence_score, inference_time)
        """
        if not self.model_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Start timing
        start_time = time.time()
        
        try:
            # Format the input following the fine-tuning format
            input_text = f"{question}\n"
            
            # Tokenize input
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            
            # Generate response with scores for confidence calculation
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    early_stopping=True
                )
            
            # Extract generated sequence
            generated_ids = outputs.sequences[0]
            
            # Decode the full response
            full_response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Extract only the generated part (after the input)
            generated_answer = full_response[len(input_text):].strip()
            
            # Stop at the first sentence or reasonable length
            if generated_answer:
                # Clean up the answer
                sentences = generated_answer.split('.')
                if len(sentences) > 1 and sentences[0].strip():
                    generated_answer = sentences[0].strip() + '.'
                else:
                    generated_answer = generated_answer.split('\n')[0].strip()
            
            # Calculate confidence score from generation probabilities
            confidence_score = self._calculate_confidence(outputs, input_ids.shape[1])
            
            # Calculate inference time
            inference_time = time.time() - start_time
            
            return generated_answer, confidence_score, inference_time
            
        except Exception as e:
            inference_time = time.time() - start_time
            print(f"Error during generation: {str(e)}")
            return f"Error generating response: {str(e)}", 0.0, inference_time
    
    def _calculate_confidence(self, outputs, input_length):
        """Calculate confidence score from generation probabilities"""
        try:
            if hasattr(outputs, 'scores') and outputs.scores:
                # Get probabilities for generated tokens
                scores = torch.stack(outputs.scores, dim=1)  # [batch_size, seq_len, vocab_size]
                probs = torch.softmax(scores, dim=-1)
                
                # Get the generated token IDs (excluding input)
                generated_tokens = outputs.sequences[0][input_length:]
                
                # Calculate probability of each generated token
                token_probs = []
                for i, token_id in enumerate(generated_tokens):
                    if i < len(outputs.scores):
                        prob = probs[0, i, token_id].item()
                        token_probs.append(prob)
                
                if token_probs:
                    # Use geometric mean for confidence (more conservative)
                    confidence = np.exp(np.mean(np.log(token_probs + 1e-10)))
                    return min(confidence, 0.99)  # Cap at 0.99
                else:
                    return 0.5
            else:
                # Fallback confidence score
                return 0.7
                
        except Exception as e:
            print(f"Error calculating confidence: {str(e)}")
            return 0.5
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.model_loaded
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.model_loaded:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": self.model_path,
            "model_type": "Fine-tuned GPT-2",
            "parameters": self.model.num_parameters() if self.model else "unknown",
            "tokenizer_vocab_size": len(self.tokenizer) if self.tokenizer else "unknown"
        }

def load_finetuned_model():
    """Convenience function to load the fine-tuned model"""
    handler = FineTunedModelHandler()
    success = handler.load_model()
    return handler if success else None

def generate_finetuned_response(question, model_handler):
    """
    Convenience function to generate response using fine-tuned model
    
    Args:
        question (str): The input question
        model_handler: The loaded model handler
        
    Returns:
        tuple: (answer, confidence_score, inference_time)
    """
    if model_handler is None or not model_handler.is_loaded():
        return "Model not loaded", 0.0, 0.0
    
    return model_handler.generate_response(question)