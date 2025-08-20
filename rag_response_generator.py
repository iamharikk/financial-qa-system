import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Any
from multistage_retriever import MultiStageRetriever

class RAGResponseGenerator:
    """RAG-enhanced response generator that uses retrieved passages to generate answers"""
    
    def __init__(self, multistage_retriever: MultiStageRetriever, model_name: str = "gpt2"):
        """
        Initialize RAG response generator
        
        Args:
            multistage_retriever: Initialized MultiStageRetriever
            model_name: Generative model to use (gpt2, distilgpt2)
        """
        self.retriever = multistage_retriever
        self.model_name = model_name
        
        # Load generative model
        print(f"Loading generative model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Model context window (GPT-2 has 1024 tokens)
        self.max_context_length = 1024
        print(f"Model loaded with context window: {self.max_context_length} tokens")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer"""
        return len(self.tokenizer.encode(text))
    
    def create_rag_prompt(self, query: str, retrieved_passages: List[Dict[str, Any]], 
                         max_context_tokens: int = 800) -> str:
        """
        Create RAG prompt by concatenating query + retrieved passages
        
        Args:
            query: User question
            retrieved_passages: Results from multistage retrieval
            max_context_tokens: Maximum tokens to use for context (leave room for generation)
        """
        print(f"\n=== CREATING RAG PROMPT ===")
        print(f"Query: '{query}'")
        print(f"Retrieved passages: {len(retrieved_passages)}")
        print(f"Max context tokens: {max_context_tokens}")
        
        # Base prompt template
        prompt_template = """You are a helpful assistant. Use the provided context to answer the question accurately. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query}
Answer:"""
        
        # Start building context from retrieved passages
        context_parts = []
        current_tokens = 0
        
        # Count tokens for base prompt (without context)
        base_prompt = prompt_template.format(context="", query=query)
        base_tokens = self.count_tokens(base_prompt)
        
        # Reserve tokens for the answer generation
        available_tokens = max_context_tokens - base_tokens
        
        print(f"Base prompt tokens: {base_tokens}")
        print(f"Available tokens for context: {available_tokens}")
        
        # Add passages until we hit token limit
        for i, passage in enumerate(retrieved_passages):
            passage_text = passage['chunk']['text']
            passage_tokens = self.count_tokens(f"Passage {i+1}: {passage_text}\n\n")
            
            if current_tokens + passage_tokens <= available_tokens:
                context_parts.append(f"Passage {i+1}: {passage_text}")
                current_tokens += passage_tokens
                print(f"Added passage {i+1}: {passage_tokens} tokens (total: {current_tokens})")
            else:
                print(f"Skipped passage {i+1}: would exceed token limit ({passage_tokens} tokens)")
                break
        
        # Create final context
        if context_parts:
            context = "\n\n".join(context_parts)
        else:
            context = "No relevant context found."
        
        final_prompt = prompt_template.format(context=context, query=query)
        final_tokens = self.count_tokens(final_prompt)
        
        print(f"Final prompt tokens: {final_tokens}")
        print(f"Used {len(context_parts)} out of {len(retrieved_passages)} passages")
        
        return final_prompt
    
    def generate_rag_response(self, query: str, final_k: int = 5, broad_k: int = 15) -> Dict[str, Any]:
        """
        Complete RAG pipeline: Retrieve + Generate
        
        Args:
            query: User question
            final_k: Number of final passages to use for generation
            broad_k: Number of candidates for stage 1 retrieval
        """
        print("=" * 80)
        print("RAG RESPONSE GENERATION PIPELINE")
        print("=" * 80)
        print(f"Query: '{query}'")
        
        # Step 1: Retrieve relevant passages using multistage retrieval
        print(f"\nStep 1: Retrieving relevant passages...")
        retrieval_results = self.retriever.retrieve(
            query=query,
            final_k=final_k,
            broad_k=broad_k
        )
        
        retrieved_passages = retrieval_results['final_results']
        
        if not retrieved_passages:
            return {
                'query': query,
                'retrieved_passages': [],
                'generated_response': "I couldn't find relevant information to answer your question.",
                'confidence_score': 0.0,
                'retrieval_stats': retrieval_results['pipeline_stats']
            }
        
        # Step 2: Create RAG prompt with retrieved context
        print(f"\nStep 2: Creating RAG prompt...")
        rag_prompt = self.create_rag_prompt(query, retrieved_passages)
        
        # Step 3: Generate response using the prompt
        print(f"\nStep 3: Generating response...")
        generated_response, confidence_score = self.generate_response_with_context(rag_prompt)
        
        return {
            'query': query,
            'retrieved_passages': retrieved_passages,
            'rag_prompt': rag_prompt,
            'generated_response': generated_response,
            'confidence_score': confidence_score,
            'retrieval_stats': retrieval_results['pipeline_stats'],
            'token_stats': {
                'prompt_tokens': self.count_tokens(rag_prompt),
                'response_tokens': self.count_tokens(generated_response)
            }
        }
    
    def generate_response_with_context(self, prompt: str) -> Tuple[str, float]:
        """Generate response using the RAG prompt with retrieved context"""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            prompt_length = len(inputs[0])
            
            # Calculate max new tokens (leave room in context window)
            max_new_tokens = min(150, self.max_context_length - prompt_length - 50)
            
            print(f"Prompt tokens: {prompt_length}")
            print(f"Max new tokens: {max_new_tokens}")
            
            # Generate with output scores for confidence calculation
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,  # Slightly lower for more focused responses
                    top_p=0.9,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode generated text
            generated_tokens = outputs.sequences[0][len(inputs[0]):]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Calculate confidence from token probabilities
            confidence_score = self.calculate_confidence_from_scores(outputs.scores)
            
            # Clean up the response
            generated_text = self.clean_response(generated_text)
            
            print(f"Generated response: {len(generated_text)} characters")
            
            return generated_text, confidence_score
            
        except Exception as e:
            print(f"Error in generation: {str(e)}")
            return f"Error generating response: {str(e)}", 0.0
    
    def clean_response(self, text: str) -> str:
        """Clean up generated response"""
        if not text:
            return "I don't have enough information to answer that question."
        
        # Remove any leftover prompt text
        if "Context:" in text or "Question:" in text or "Answer:" in text:
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line and not any(marker in line for marker in ["Context:", "Question:", "Answer:", "Passage"]):
                    text = line
                    break
        
        # Remove repetitions
        text = self.remove_repetitions(text)
        
        # Truncate at natural sentence ending
        text = self.truncate_at_sentence_end(text)
        
        return text.strip()
    
    def calculate_confidence_from_scores(self, scores) -> float:
        """Calculate confidence score from model output probabilities"""
        if not scores:
            return 0.5
        
        try:
            all_probs = []
            for score in scores:
                probs = F.softmax(score, dim=-1)
                max_prob = torch.max(probs).item()
                all_probs.append(max_prob)
            
            avg_confidence = np.mean(all_probs)
            scaled_confidence = 0.3 + (avg_confidence * 0.6)
            
            return round(float(scaled_confidence), 3)
            
        except Exception:
            return 0.5
    
    def remove_repetitions(self, text: str) -> str:
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
                break
        
        return '. '.join(unique_sentences)
    
    def truncate_at_sentence_end(self, text: str) -> str:
        """Truncate text at the last complete sentence"""
        if not text:
            return text
        
        last_period = text.rfind('.')
        last_exclamation = text.rfind('!')
        last_question = text.rfind('?')
        
        last_punct = max(last_period, last_exclamation, last_question)
        
        if last_punct > 0 and last_punct < len(text) - 1:
            return text[:last_punct + 1].strip()
        
        return text.strip()

# Example usage:
if __name__ == "__main__":
    from text_chunker import SimpleTextChunker
    from embedding_indexer import SimpleEmbeddingIndexer
    
    # Load and index data
    print("Setting up RAG system...")
    chunker = SimpleTextChunker()
    chunks = chunker.process_file("data/financial-statements/tcs_clean.txt")
    
    indexer = SimpleEmbeddingIndexer()
    indexer.index_chunks(chunks['small'][:50])
    
    retriever = MultiStageRetriever(indexer)
    
    # Create RAG response generator
    rag_generator = RAGResponseGenerator(retriever)
    
    # Test questions
    test_queries = [
        "What is TCS's revenue?",
        "How did the company perform financially?",
        "What are the main assets of TCS?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*100}")
        print(f"TESTING RAG RESPONSE: '{query}'")
        print(f"{'='*100}")
        
        result = rag_generator.generate_rag_response(query, final_k=3, broad_k=8)
        
        print(f"\nRETRIEVED PASSAGES ({len(result['retrieved_passages'])}):")
        for i, passage in enumerate(result['retrieved_passages'], 1):
            print(f"{i}. Score: {passage['cross_encoder_score']:.3f}")
            print(f"   {passage['chunk']['text'][:100]}...")
        
        print(f"\nGENERATED RESPONSE:")
        print(f"Confidence: {result['confidence_score']:.3f}")
        print(f"Response: {result['generated_response']}")
        
        print(f"\nSTATS:")
        print(f"Prompt tokens: {result['token_stats']['prompt_tokens']}")
        print(f"Response tokens: {result['token_stats']['response_tokens']}")
        print(f"Total time: {result['retrieval_stats']['total_time']:.2f}s")