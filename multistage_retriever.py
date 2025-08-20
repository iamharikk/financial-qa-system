from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from hybrid_retriever import HybridRetriever
from embedding_indexer import SimpleEmbeddingIndexer
import time

class MultiStageRetriever:
    """Advanced RAG with multi-stage retrieval: Broad retrieval + Cross-encoder reranking"""
    
    def __init__(self, indexer: SimpleEmbeddingIndexer, 
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize multi-stage retriever
        
        Args:
            indexer: Your existing embedding indexer
            cross_encoder_model: Cross-encoder for reranking (smaller, faster model)
        """
        # Stage 1: Broad retrieval using existing hybrid approach
        self.hybrid_retriever = HybridRetriever(indexer)
        
        # Stage 2: Precise reranking with cross-encoder
        print(f"Loading cross-encoder model: {cross_encoder_model}")
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        print("Cross-encoder model loaded!")
    
    def stage1_broad_retrieval(self, query: str, broad_k: int = 20) -> List[Dict[str, Any]]:
        """
        Stage 1: Cast a wide net to get many potentially relevant candidates
        Uses bi-encoder (fast) to get initial candidates
        """
        print(f"\n=== STAGE 1: BROAD RETRIEVAL (Bi-Encoder) ===")
        print(f"Retrieving top-{broad_k} candidates using hybrid search...")
        
        start_time = time.time()
        
        # Use existing hybrid retrieval to get broad candidates
        results = self.hybrid_retriever.retrieve(
            query=query,
            top_k=broad_k,           # Get many candidates
            dense_k=broad_k + 5,     # Search more dense results
            sparse_k=broad_k + 5,    # Search more sparse results
            alpha=0.6                # Balanced weighting
        )
        
        candidates = results['final_results']
        stage1_time = time.time() - start_time
        
        print(f"Stage 1 complete: Retrieved {len(candidates)} candidates in {stage1_time:.2f}s")
        
        # Show top few candidates
        print("Top 3 broad candidates from bi-encoder:")
        for i, candidate in enumerate(candidates[:3], 1):
            print(f"  {i}. Score: {candidate['combined_score']:.3f} - {candidate['chunk']['text'][:60]}...")
        
        return candidates
    
    def stage2_cross_encoder_rerank(self, query: str, candidates: List[Dict[str, Any]], 
                                   final_k: int = 5) -> List[Dict[str, Any]]:
        """
        Stage 2: Use cross-encoder to precisely rerank candidates
        Cross-encoder sees query+document together for better accuracy
        """
        print(f"\n=== STAGE 2: CROSS-ENCODER RERANKING ===")
        print(f"Reranking {len(candidates)} candidates using cross-encoder...")
        print("Cross-encoder processes [QUERY] + [DOCUMENT] together for precise relevance")
        
        if not candidates:
            return []
        
        start_time = time.time()
        
        # Prepare query-document pairs for cross-encoder
        query_doc_pairs = []
        for candidate in candidates:
            # Cross-encoder takes [query, document] pairs
            query_doc_pairs.append([query, candidate['chunk']['text']])
        
        print(f"Creating {len(query_doc_pairs)} query-document pairs...")
        
        # Get cross-encoder scores (these are more accurate relevance scores)
        print("Computing cross-encoder relevance scores...")
        cross_encoder_scores = self.cross_encoder.predict(query_doc_pairs)
        
        print("Cross-encoder scoring complete. Reranking results...")
        
        # Combine cross-encoder scores with original results
        reranked_results = []
        for i, (candidate, ce_score) in enumerate(zip(candidates, cross_encoder_scores)):
            reranked_result = {
                'chunk': candidate['chunk'],
                'cross_encoder_score': float(ce_score),      # New precise score
                'stage1_score': candidate['combined_score'],  # Original hybrid score
                'stage1_rank': candidate.get('final_rank', i + 1),
                'dense_score': candidate.get('dense_score', 0),
                'sparse_score': candidate.get('sparse_score', 0),
                'search_type': 'multistage'
            }
            reranked_results.append(reranked_result)
        
        # Sort by cross-encoder score (most important)
        reranked_results.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
        
        # Add final ranks
        for i, result in enumerate(reranked_results):
            result['final_rank'] = i + 1
        
        stage2_time = time.time() - start_time
        print(f"Stage 2 complete: Reranked in {stage2_time:.2f}s")
        
        # Show reranking changes
        print(f"\nReranking Results (top {min(final_k, len(reranked_results))}):")
        print("Format: [Final Rank]: CE Score (was Stage1 Rank)")
        for result in reranked_results[:final_k]:
            rank_change = result['final_rank'] - result['stage1_rank']
            change_symbol = "↑" if rank_change < 0 else "↓" if rank_change > 0 else "="
            print(f"  Rank {result['final_rank']}: CE Score: {result['cross_encoder_score']:.3f} "
                  f"(was rank {result['stage1_rank']}) {change_symbol}")
            print(f"    Text: {result['chunk']['text'][:80]}...")
        
        return reranked_results[:final_k]
    
    def retrieve(self, query: str, final_k: int = 5, broad_k: int = 20) -> Dict[str, Any]:
        """
        Complete multi-stage retrieval pipeline
        
        Args:
            query: User query
            final_k: Number of final results
            broad_k: Number of candidates for stage 1 (should be 3-4x final_k)
        """
        print("=" * 80)
        print(f"MULTI-STAGE RETRIEVAL PIPELINE")
        print("=" * 80)
        print(f"Query: '{query}'")
        print(f"Pipeline: Bi-encoder broad search ({broad_k}) -> Cross-encoder rerank -> Top {final_k}")
        
        overall_start = time.time()
        
        # Stage 1: Broad retrieval with bi-encoder
        candidates = self.stage1_broad_retrieval(query, broad_k=broad_k)
        
        # Stage 2: Cross-encoder reranking
        final_results = self.stage2_cross_encoder_rerank(query, candidates, final_k=final_k)
        
        total_time = time.time() - overall_start
        
        print(f"\nMULTI-STAGE RETRIEVAL COMPLETE")
        print(f"Total time: {total_time:.2f}s")
        print(f"Final results: {len(final_results)}")
        
        return {
            'query': query,
            'stage1_candidates': candidates,
            'final_results': final_results,
            'pipeline_stats': {
                'broad_k': broad_k,
                'final_k': final_k,
                'total_time': total_time,
                'stage1_count': len(candidates),
                'final_count': len(final_results)
            }
        }

# Example usage and comparison:
if __name__ == "__main__":
    from text_chunker import SimpleTextChunker
    
    # Load and index data
    print("Loading and indexing data...")
    chunker = SimpleTextChunker()
    chunks = chunker.process_file("data/financial-statements/tcs_clean.txt")
    
    indexer = SimpleEmbeddingIndexer()
    indexer.index_chunks(chunks['small'][:50])  # Use subset for demo
    
    # Create multi-stage retriever
    multistage_retriever = MultiStageRetriever(indexer)
    
    # Test queries
    test_queries = [
        "What is TCS revenue and profit?",
        "company financial performance",
        "balance sheet assets and liabilities"
    ]
    
    for query in test_queries:
        print(f"\n{'='*100}")
        print(f"TESTING: '{query}'")
        print(f"{'='*100}")
        
        # Multi-stage retrieval
        results = multistage_retriever.retrieve(
            query=query,
            final_k=3,      # Final results
            broad_k=10      # Stage 1 candidates
        )
        
        print(f"\nFINAL RANKING COMPARISON:")
        print("Shows how cross-encoder changed the ranking from stage 1")
        for result in results['final_results']:
            print(f"\nRank {result['final_rank']}:")
            print(f"   Cross-Encoder Score: {result['cross_encoder_score']:.3f}")
            print(f"   Stage 1 Rank: {result['stage1_rank']} (Bi-encoder Score: {result['stage1_score']:.3f})")
            print(f"   Text: {result['chunk']['text'][:120]}...")
        
        print(f"\nPerformance: {results['pipeline_stats']['total_time']:.2f}s total")
        print(f"   Processed {results['pipeline_stats']['stage1_count']} -> {results['pipeline_stats']['final_count']} results")