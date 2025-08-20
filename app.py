import streamlit as st
import time
from guardrails import validate_all_inputs
from model_handler import load_gpt2_small_model, generate_response
from text_chunker import SimpleTextChunker
from embedding_indexer import SimpleEmbeddingIndexer
from multistage_retriever import MultiStageRetriever
from rag_response_generator import RAGResponseGenerator

st.title("Financial QA System App")

# Load model with caching
@st.cache_resource
def get_model():
    return load_gpt2_small_model()

# Load RAG system with caching
@st.cache_resource
def get_rag_system():
    """Initialize the complete RAG system"""
    try:
        with st.spinner("Setting up RAG system..."):
            # Load and chunk data (use QA pairs for better responses)
            chunker = SimpleTextChunker()
            chunks = chunker.process_file("data/q-and-a/qa-pairs.txt")
            
            # Create embeddings and indices
            indexer = SimpleEmbeddingIndexer()
            indexer.index_chunks(chunks['small'][:100])  # Use more chunks for better results
            
            # Create multistage retriever
            retriever = MultiStageRetriever(indexer)
            
            # Create RAG response generator
            rag_generator = RAGResponseGenerator(retriever)
            
            return rag_generator
    except Exception as e:
        st.error(f"Error setting up RAG system: {str(e)}")
        return None

# Display model loading status
with st.spinner("Loading GPT-2 Small model..."):
    model = get_model()

if model is not None:
    st.success("Model loaded successfully!")
else:
    st.error("Failed to load model")

# Load RAG system
rag_system = get_rag_system()

if rag_system is not None:
    st.success("RAG system initialized successfully!")
else:
    st.error("Failed to initialize RAG system")

method = st.radio(
    "Select the method:",
    ["Retrieval-Augmented Generation (RAG) System Implementation", "Fine-Tuned Model System Implementation"]
)

text = st.text_area("Enter your query:", height=100)

if st.button("Ask"):
    if text and method:
        # Validate inputs using guardrails
        is_valid, error_message = validate_all_inputs(text, method)
        
        if not is_valid:
            st.error(f"Input validation failed: {error_message}")
        else:
            if model is None:
                st.error("Model not loaded. Please refresh the page.")
            else:
                # Start timing the response
                start_time = time.time()
                
                if method == "Retrieval-Augmented Generation (RAG) System Implementation":
                    # Use RAG system
                    if rag_system is None:
                        st.error("RAG system not loaded. Please refresh the page.")
                    else:
                        with st.spinner("Processing with RAG system..."):
                            rag_result = rag_system.generate_rag_response(text, final_k=3, broad_k=10)
                        
                        answer = rag_result['generated_response']
                        confidence_score = rag_result['confidence_score']
                        
                        # Calculate actual response time
                        end_time = time.time()
                        response_time = round(end_time - start_time, 3)
                        
                        st.success("Query processed successfully with RAG!")
                        st.write(f"**Answer:** {answer}")
                        st.write(f"**Confidence Score:** {confidence_score}")
                        st.write(f"**Method Used:** {method}")
                        st.write(f"**Response Time:** {response_time} seconds")
                        
                        # Show RAG details
                        with st.expander("RAG System Details"):
                            st.write(f"**Retrieved Passages:** {len(rag_result['retrieved_passages'])}")
                            
                            for i, passage in enumerate(rag_result['retrieved_passages'], 1):
                                st.write(f"**Passage {i}** (Score: {passage['cross_encoder_score']:.3f}):")
                                st.write(f"{passage['chunk']['text'][:200]}...")
                                st.write("---")
                            
                            st.write(f"**Token Usage:**")
                            st.write(f"- Prompt tokens: {rag_result['token_stats']['prompt_tokens']}")
                            st.write(f"- Response tokens: {rag_result['token_stats']['response_tokens']}")
                            
                            st.write(f"**Retrieval Stats:**")
                            st.write(f"- Stage 1 candidates: {rag_result['retrieval_stats']['stage1_count']}")
                            st.write(f"- Final results: {rag_result['retrieval_stats']['final_count']}")
                            st.write(f"- Retrieval time: {rag_result['retrieval_stats']['total_time']:.2f}s")
                
                else:
                    # Use fine-tuned model (original implementation)
                    with st.spinner("Generating response..."):
                        answer, confidence_score = generate_response(text, model)
                    
                    # Calculate actual response time
                    end_time = time.time()
                    response_time = round(end_time - start_time, 3)
                
                    st.success("Query processed successfully!")
                    st.write(f"**Answer:** {answer}")
                    st.write(f"**Confidence Score:** {confidence_score}")
                    st.write(f"**Method Used:** {method}")
                    st.write(f"**Response Time:** {response_time} seconds")
    else:
        if not text:
            st.warning("Please enter some text first!")
        if not method:
            st.warning("Please select a method first!")