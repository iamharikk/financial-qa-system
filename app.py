import streamlit as st
import time
from guardrails import validate_all_inputs
from model_handler import load_distilgpt2_model, generate_response

st.title("Financial QA System App")

# Load model with caching
@st.cache_resource
def get_model():
    return load_distilgpt2_model()

# Display model loading status
with st.spinner("Loading DistilGPT-2 model..."):
    model = get_model()

if model is not None:
    st.success("✅ Model loaded successfully!")
else:
    st.error("❌ Failed to load model")

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
                
                # Generate response using DistilGPT-2
                with st.spinner("Generating response..."):
                    answer, confidence_score = generate_response(text, model)
                
                method_used = method
                
                # Calculate actual response time
                end_time = time.time()
                response_time = round(end_time - start_time, 3)
            
            st.success("Query processed successfully!")
            st.write(f"**Answer:** {answer}")
            st.write(f"**Confidence Score:** {confidence_score}")
            st.write(f"**Method Used:** {method_used}")
            st.write(f"**Response Time:** {response_time} seconds")
    else:
        if not text:
            st.warning("Please enter some text first!")
        if not method:
            st.warning("Please select a method first!")