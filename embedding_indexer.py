from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any
import os

class SimpleEmbeddingIndexer:
    """Simple embedding and indexing system for RAG"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Dense vector store (FAISS)
        self.faiss_index = None
        self.chunk_metadata = []
        
        # Sparse index (TF-IDF)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.chunk_texts = []
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """Convert text chunks to dense embeddings"""
        texts = [chunk['text'] for chunk in chunks]
        print(f"Creating embeddings for {len(texts)} chunks...")
        
        embeddings = self.embedding_model.encode(
            texts, 
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index for dense vector search"""
        print(f"Building FAISS index with {len(embeddings)} vectors...")
        
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings.astype(np.float32))
        
        print(f"FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def build_sparse_index(self, texts: List[str]):
        """Build TF-IDF sparse index for keyword search"""
        print(f"Building TF-IDF index for {len(texts)} texts...")
        
        self.chunk_texts = texts
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        print(f"TF-IDF index built with vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
    
    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Index chunks for both dense and sparse retrieval"""
        if not chunks:
            print("No chunks to index")
            return
        
        print(f"Indexing {len(chunks)} chunks...")
        
        self.chunk_metadata = chunks
        
        embeddings = self.create_embeddings(chunks)
        self.build_faiss_index(embeddings)
        
        texts = [chunk['text'] for chunk in chunks]
        self.build_sparse_index(texts)
        
        print("Indexing complete!")
    
    def dense_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using dense embeddings (semantic similarity)"""
        if self.faiss_index is None:
            return []
        
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.faiss_index.search(query_embedding.astype(np.float32), top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:
                result = {
                    'chunk': self.chunk_metadata[idx],
                    'score': float(score),
                    'rank': i + 1,
                    'search_type': 'dense'
                }
                results.append(result)
        
        return results
    
    def sparse_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using TF-IDF sparse vectors (keyword matching)"""
        if self.tfidf_matrix is None:
            return []
        
        query_vector = self.tfidf_vectorizer.transform([query])
        scores = (self.tfidf_matrix * query_vector.T).toarray().flatten()
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            if scores[idx] > 0:
                result = {
                    'chunk': self.chunk_metadata[idx],
                    'score': float(scores[idx]),
                    'rank': i + 1,
                    'search_type': 'sparse'
                }
                results.append(result)
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.7) -> List[Dict[str, Any]]:
        """Combine dense and sparse search results"""
        dense_results = self.dense_search(query, top_k * 2)
        sparse_results = self.sparse_search(query, top_k * 2)
        
        # Combine and rerank
        combined_scores = {}
        
        for result in dense_results:
            chunk_id = result['chunk']['id']
            combined_scores[chunk_id] = alpha * result['score']
        
        for result in sparse_results:
            chunk_id = result['chunk']['id']
            if chunk_id in combined_scores:
                combined_scores[chunk_id] += (1 - alpha) * result['score']
            else:
                combined_scores[chunk_id] = (1 - alpha) * result['score']
        
        # Sort by combined score
        sorted_chunks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for i, (chunk_id, score) in enumerate(sorted_chunks[:top_k]):
            # Find the chunk by ID
            chunk = next(c for c in self.chunk_metadata if c['id'] == chunk_id)
            result = {
                'chunk': chunk,
                'score': float(score),
                'rank': i + 1,
                'search_type': 'hybrid'
            }
            final_results.append(result)
        
        return final_results

# Example usage:
if __name__ == "__main__":
    from text_chunker import SimpleTextChunker
    
    chunker = SimpleTextChunker()
    chunks = chunker.process_file("data/financial-statements/tcs_clean.txt")
    
    small_chunks = chunks['small'][:50]
    
    indexer = SimpleEmbeddingIndexer()
    indexer.index_chunks(small_chunks)
    
    query = "revenue and profit"
    
    print("\nHybrid search results:")
    hybrid_results = indexer.hybrid_search(query, top_k=3)
    for result in hybrid_results:
        print(f"Score: {result['score']:.3f} - {result['chunk']['text'][:100]}...")