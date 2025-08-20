import re
import string
from typing import Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np

# Download required NLTK data (run once)
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download all required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) 
nltk.download('stopwords', quiet=True)

class QueryPreprocessor:
    """Simple query preprocessing and embedding for retrieval"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.stop_words = set(stopwords.words('english'))
        
        # Keep important financial terms that might be stopwords
        financial_terms = {'will', 'would', 'should', 'can', 'may', 'must', 'no', 'not'}
        self.stop_words = self.stop_words - financial_terms
        
        # Load embedding model
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        print("Embedding model loaded!")
    
    def step1_clean_text(self, text: str) -> str:
        """Step 1: Basic text cleaning"""
        print(f"Original: '{text}'")
        
        # Convert to lowercase
        text = text.lower()
        print(f"Lowercase: '{text}'")
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        print(f"Remove special chars: '{text}'")
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        print(f"Clean whitespace: '{text}'")
        
        return text
    
    def step2_remove_stopwords(self, text: str) -> str:
        """Step 2: Remove stopwords"""
        print(f"Input: '{text}'")
        
        # Tokenize text
        tokens = word_tokenize(text)
        print(f"Tokens: {tokens}")
        
        # Filter out stopwords
        filtered_tokens = []
        removed_words = []
        
        for word in tokens:
            if word in self.stop_words:
                removed_words.append(word)
            else:
                filtered_tokens.append(word)
        
        print(f"Removed stopwords: {removed_words}")
        print(f"Kept words: {filtered_tokens}")
        
        result = ' '.join(filtered_tokens)
        print(f"Result: '{result}'")
        
        return result
    
    def step3_generate_embedding(self, text: str) -> np.ndarray:
        """Step 3: Generate dense embedding for semantic search"""
        print(f"Input text: '{text}'")
        
        # Generate embedding
        embedding = self.embedding_model.encode([text], convert_to_numpy=True)
        
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding dimension: {embedding.shape[1]}")
        print(f"First 5 values: {embedding[0][:5]}")
        
        return embedding[0]  # Return single embedding vector
    
    def preprocess_query(self, query: str) -> Dict:
        """Complete preprocessing with step-by-step output"""
        print("=" * 60)
        print(f"PREPROCESSING QUERY: '{query}'")
        print("=" * 60)
        
        print("\nSTEP 1: TEXT CLEANING")
        print("-" * 30)
        cleaned = self.step1_clean_text(query)
        
        print("\nSTEP 2: STOPWORD REMOVAL")
        print("-" * 30)
        no_stopwords = self.step2_remove_stopwords(cleaned)
        
        print("\nSTEP 3: GENERATE EMBEDDING")
        print("-" * 30)
        # Use cleaned query for embedding (preserves meaning better than no_stopwords)
        embedding = self.step3_generate_embedding(cleaned)
        
        print("\nFINAL RESULTS:")
        print("-" * 30)
        results = {
            'original': query,
            'cleaned': cleaned,
            'no_stopwords': no_stopwords,
            'embedding': embedding
        }
        
        # Print text results (not the full embedding)
        for key, value in results.items():
            if key != 'embedding':
                print(f"{key}: '{value}'")
            else:
                print(f"{key}: vector of shape {value.shape}")
        
        return results

# Example usage:
if __name__ == "__main__":
    preprocessor = QueryPreprocessor()
    
    # Test with different query types
    test_queries = [
        "What is TCS's revenue for 2023?",
        "Show me the company's profit & loss statement",
        "How much did they earn last year?"
    ]
    
    for query in test_queries:
        result = preprocessor.preprocess_query(query)
        print("\n" + "="*60 + "\n")