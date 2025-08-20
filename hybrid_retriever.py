from typing import List, Dict, Any
import numpy as np
from query_preprocessor import QueryPreprocessor
from embedding_indexer import SimpleEmbeddingIndexer

class HybridRetriever:
    """Complete hybrid retrieval pipeline"""
    
    def __init__(self, indexer: SimpleEmbeddingIndexer):
        self.indexer = indexer
        self.preprocessor = QueryPreprocessor()
    
    def dense_retrieval(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Step 1: Dense retrieval using query embedding"""
        print(f"\n=== DENSE RETRIEVAL (Semantic Search) ===")
        print(f"Query embedding shape: {query_embedding.shape}")
        print(f"Searching for top-{top_k} semantically similar chunks...")
        
        # Use the indexer's search method but with our embedding
        import faiss
        if self.indexer.faiss_index is None:
            print("No FAISS index found!")
            return []
        
        # Normalize and search
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.indexer.faiss_index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid result
                result = {
                    'chunk': self.indexer.chunk_metadata[idx],
                    'score': float(score),
                    'rank': i + 1,
                    'search_type': 'dense'
                }
                results.append(result)
                print(f"  Rank {i+1}: Score {score:.3f} - {result['chunk']['text'][:60]}...")
        
        print(f"Dense retrieval found {len(results)} results")
        return results
    
    def sparse_retrieval(self, processed_query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Step 2: Sparse retrieval using processed query text"""
        print(f"\n=== SPARSE RETRIEVAL (Keyword Search) ===")
        print(f"Processed query: '{processed_query}'")
        print(f"Searching for top-{top_k} keyword matches...")
        
        if self.indexer.tfidf_matrix is None:
            print("No TF-IDF index found!")
            return []
        
        # Transform query to TF-IDF vector
        query_vector = self.indexer.tfidf_vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        scores = (self.indexer.tfidf_matrix * query_vector.T).toarray().flatten()
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            if scores[idx] > 0:  # Only positive scores
                result = {
                    'chunk': self.indexer.chunk_metadata[idx],
                    'score': float(scores[idx]),
                    'rank': i + 1,
                    'search_type': 'sparse'
                }
                results.append(result)
                print(f"  Rank {i+1}: Score {scores[idx]:.3f} - {result['chunk']['text'][:60]}...")
        
        print(f"Sparse retrieval found {len(results)} results")
        return results
    
    def combine_results(self, dense_results: List[Dict], sparse_results: List[Dict], 
                       final_top_k: int = 5, alpha: float = 0.7) -> List[Dict[str, Any]]:
        """Step 3: Combine results using weighted score fusion"""
        print(f"\n=== COMBINING RESULTS (Weighted Fusion) ===")
        print(f"Alpha = {alpha} (dense weight), Beta = {1-alpha} (sparse weight)")
        print(f"Combining {len(dense_results)} dense + {len(sparse_results)} sparse results")
        
        # Collect all unique chunks with combined scores
        combined_scores = {}
        
        # Add dense results
        for result in dense_results:
            chunk_id = result['chunk']['id']
            combined_scores[chunk_id] = {
                'chunk': result['chunk'],
                'dense_score': result['score'],
                'sparse_score': 0.0,
                'dense_rank': result['rank'],
                'sparse_rank': None
            }
        
        # Add sparse results
        for result in sparse_results:
            chunk_id = result['chunk']['id']
            if chunk_id in combined_scores:
                # Chunk found in both searches
                combined_scores[chunk_id]['sparse_score'] = result['score']
                combined_scores[chunk_id]['sparse_rank'] = result['rank']
                print(f"  Overlap found: {result['chunk']['text'][:40]}...")
            else:
                # Chunk only in sparse results
                combined_scores[chunk_id] = {
                    'chunk': result['chunk'],
                    'dense_score': 0.0,
                    'sparse_score': result['score'],
                    'dense_rank': None,
                    'sparse_rank': result['rank']
                }
        
        # Calculate combined scores
        final_results = []
        for chunk_id, data in combined_scores.items():
            combined_score = (alpha * data['dense_score'] + 
                            (1 - alpha) * data['sparse_score'])
            
            result = {
                'chunk': data['chunk'],
                'combined_score': combined_score,
                'dense_score': data['dense_score'],
                'sparse_score': data['sparse_score'],
                'dense_rank': data['dense_rank'],
                'sparse_rank': data['sparse_rank'],
                'search_type': 'hybrid'
            }
            final_results.append(result)
        
        # Sort by combined score
        final_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Add final ranks and return top-k
        for i, result in enumerate(final_results[:final_top_k]):
            result['final_rank'] = i + 1
            print(f"  Final Rank {i+1}: Combined={result['combined_score']:.3f} "
                  f"(Dense={result['dense_score']:.3f}, Sparse={result['sparse_score']:.3f})")
            print(f"    Text: {result['chunk']['text'][:60]}...")
        
        print(f"Final hybrid results: {len(final_results[:final_top_k])} chunks")
        return final_results[:final_top_k]
    
    def retrieve(self, query: str, top_k: int = 5, dense_k: int = 10, sparse_k: int = 10,
                alpha: float = 0.7) -> Dict[str, Any]:
        """Complete hybrid retrieval pipeline"""
        print("=" * 80)
        print(f"HYBRID RETRIEVAL PIPELINE")
        print("=" * 80)
        print(f"Query: '{query}'")
        print(f"Target final results: {top_k}")
        print(f"Dense retrieval: top-{dense_k}, Sparse retrieval: top-{sparse_k}")
        print(f"Fusion weight: {alpha} dense + {1-alpha} sparse")
        
        # Step 0: Preprocess query
        print(f"\n=== QUERY PREPROCESSING ===")
        processed = self.preprocessor.preprocess_query(query)
        
        # Step 1: Dense retrieval
        dense_results = self.dense_retrieval(processed['embedding'], top_k=dense_k)
        
        # Step 2: Sparse retrieval  
        sparse_results = self.sparse_retrieval(processed['no_stopwords'], top_k=sparse_k)
        
        # Step 3: Combine results
        final_results = self.combine_results(dense_results, sparse_results, 
                                           final_top_k=top_k, alpha=alpha)
        
        return {
            'query': query,
            'processed_query': processed,
            'dense_results': dense_results,
            'sparse_results': sparse_results,
            'final_results': final_results,
            'retrieval_stats': {
                'dense_count': len(dense_results),
                'sparse_count': len(sparse_results),
                'final_count': len(final_results),
                'alpha': alpha
            }
        }

# Example usage:
if __name__ == "__main__":
    from text_chunker import SimpleTextChunker
    
    # Load and index data
    print("Loading and indexing data...")
    chunker = SimpleTextChunker()
    chunks = chunker.process_file("data/financial-statements/tcs_clean.txt")
    
    indexer = SimpleEmbeddingIndexer()
    indexer.index_chunks(chunks['small'][:50])  # Use subset for demo
    
    # Create retriever
    retriever = HybridRetriever(indexer)
    
    # Test retrieval
    test_query = "What is TCS revenue and profit for 2023?"
    
    results = retriever.retrieve(
        query=test_query,
        top_k=3,        # Final results
        dense_k=8,      # Dense search results
        sparse_k=8,     # Sparse search results  
        alpha=0.6       # 60% dense, 40% sparse
    )
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Query: {results['query']}")
    print(f"Total final results: {len(results['final_results'])}")
    
    for result in results['final_results']:
        print(f"\nRank {result['final_rank']}:")
        print(f"  Combined Score: {result['combined_score']:.3f}")
        print(f"  Dense: {result['dense_score']:.3f} (rank {result['dense_rank']})")
        print(f"  Sparse: {result['sparse_score']:.3f} (rank {result['sparse_rank']})")
        print(f"  Text: {result['chunk']['text'][:100]}...")