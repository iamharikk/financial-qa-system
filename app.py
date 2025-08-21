import streamlit as st
import time
from guardrails import validate_all_inputs
from model_handler import load_gpt2_small_model, generate_response
from finetuned_model_handler import load_finetuned_model, generate_finetuned_response
from text_chunker import SimpleTextChunker
from embedding_indexer import SimpleEmbeddingIndexer
from multistage_retriever import MultiStageRetriever
from rag_response_generator import RAGResponseGenerator

st.title("Financial QA System App")

# Load models with caching
@st.cache_resource
def get_model():
    return load_gpt2_small_model()

@st.cache_resource
def get_finetuned_model():
    """Load the fine-tuned model"""
    try:
        with st.spinner("Loading fine-tuned model..."):
            return load_finetuned_model()
    except Exception as e:
        st.error(f"Error loading fine-tuned model: {str(e)}")
        return None

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
    st.success("Base GPT-2 model loaded successfully!")
else:
    st.error("Failed to load base GPT-2 model")

# Load fine-tuned model
finetuned_model = get_finetuned_model()

if finetuned_model is not None:
    st.success("Fine-tuned model loaded successfully!")
    model_info = finetuned_model.get_model_info()
    st.info(f"Fine-tuned model: {model_info.get('model_type', 'Unknown')} with {model_info.get('parameters', 'unknown')} parameters")
else:
    st.error("Failed to load fine-tuned model. Make sure the model exists in './fine_tuned_sft_model' directory.")

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
                    # Use fine-tuned model
                    if finetuned_model is None:
                        st.error("Fine-tuned model not loaded. Please refresh the page.")
                    else:
                        with st.spinner("Processing with fine-tuned model..."):
                            answer, confidence_score, inference_time = generate_finetuned_response(text, finetuned_model)
                        
                        # Calculate total response time
                        end_time = time.time()
                        total_response_time = round(end_time - start_time, 3)
                        
                        st.success("Query processed successfully with fine-tuned model!")
                        st.write(f"**Answer:** {answer}")
                        st.write(f"**Confidence Score:** {confidence_score:.3f}")
                        st.write(f"**Method Used:** {method}")
                        st.write(f"**Inference Time:** {inference_time:.3f} seconds")
                        st.write(f"**Total Response Time:** {total_response_time} seconds")
                        
                        # Show fine-tuned model details
                        with st.expander("Fine-Tuned Model Details"):
                            model_info = finetuned_model.get_model_info()
                            st.write(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")
                            st.write(f"**Model Parameters:** {model_info.get('parameters', 'unknown'):,}")
                            st.write(f"**Tokenizer Vocab Size:** {model_info.get('tokenizer_vocab_size', 'unknown'):,}")
                            st.write(f"**Model Path:** {model_info.get('model_path', 'unknown')}")
                            
                            # Show confidence interpretation
                            if confidence_score >= 0.8:
                                confidence_level = "High"
                                confidence_color = "green"
                            elif confidence_score >= 0.6:
                                confidence_level = "Medium"
                                confidence_color = "orange"
                            else:
                                confidence_level = "Low"
                                confidence_color = "red"
                            
                            st.write(f"**Confidence Level:** :{confidence_color}[{confidence_level}]")
                            st.write(f"**Response Quality:** Based on token generation probabilities")
    else:
        if not text:
            st.warning("Please enter some text first!")
        if not method:
            st.warning("Please select a method first!")