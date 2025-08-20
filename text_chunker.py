from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
from typing import List, Dict, Any

class SimpleTextChunker:
    """Simple text chunker using LangChain for RAG implementation"""
    
    def __init__(self):
        # Create text splitters for two different chunk sizes
        self.small_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,  # 100 tokens approximately
            chunk_overlap=20,  # Small overlap to maintain context
            length_function=len,  # Use character length as approximation
            separators=["\n\n", "\n", " ", ""]  # Split on paragraphs, then lines, then spaces
        )
        
        self.large_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  # 400 tokens approximately
            chunk_overlap=50,  # Larger overlap for bigger chunks
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def create_chunks(self, text: str, source_file: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create chunks from text with two different sizes
        Returns: Dictionary with 'small' and 'large' chunk lists
        """
        results = {}
        
        # Create small chunks (100 tokens)
        small_texts = self.small_splitter.split_text(text)
        small_chunks = []
        
        for i, chunk_text in enumerate(small_texts):
            chunk = {
                'id': str(uuid.uuid4()),  # Unique ID for each chunk
                'text': chunk_text,
                'chunk_size': 'small',
                'chunk_index': i,
                'metadata': {
                    'source_file': source_file,
                    'chunk_number': i + 1,
                    'total_chunks': len(small_texts),
                    'character_count': len(chunk_text)
                }
            }
            small_chunks.append(chunk)
        
        # Create large chunks (400 tokens)
        large_texts = self.large_splitter.split_text(text)
        large_chunks = []
        
        for i, chunk_text in enumerate(large_texts):
            chunk = {
                'id': str(uuid.uuid4()),  # Unique ID for each chunk
                'text': chunk_text,
                'chunk_size': 'large',
                'chunk_index': i,
                'metadata': {
                    'source_file': source_file,
                    'chunk_number': i + 1,
                    'total_chunks': len(large_texts),
                    'character_count': len(chunk_text)
                }
            }
            large_chunks.append(chunk)
        
        results['small'] = small_chunks
        results['large'] = large_chunks
        
        return results
    
    def process_file(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a text file and return chunks
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self.create_chunks(text, source_file=file_path)

# Example usage:
if __name__ == "__main__":
    chunker = SimpleTextChunker()
    
    # Test with a sample file
    chunks = chunker.process_file("data/financial-statements/tcs_clean.txt")
    
    print(f"Created {len(chunks['small'])} small chunks")
    print(f"Created {len(chunks['large'])} large chunks")
    
    # Show first chunk example
    if chunks['small']:
        first_chunk = chunks['small'][0]
        print(f"\nExample chunk:")
        print(f"ID: {first_chunk['id']}")
        print(f"Text: {first_chunk['text'][:100]}...")
        print(f"Metadata: {first_chunk['metadata']}")