"""Test script to compare base GPT-2 vs fine-tuned model responses"""

from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model(model_path, model_name):
    print(f"\n{'='*50}")
    print(f"Testing {model_name}")
    print(f"{'='*50}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test questions
    questions = [
        "What was TCS's sales turnover in Mar '25?",
        "What was TCS's net profit in Mar '25?",
        "What was TCS's total income in Mar '25?"
    ]
    
    for question in questions:
        input_text = f"Question: {question}\nAnswer: "
        
        inputs = tokenizer(input_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 50,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response[len(input_text):].strip().split('\n')[0]
        
        print(f"Q: {question}")
        print(f"A: {answer}")
        print("-" * 40)

if __name__ == "__main__":
    import torch
    
    # Test base GPT-2
    test_model("gpt2", "Base GPT-2")
    
    # Test fine-tuned model
    test_model("./fine_tuned_sft_model", "Fine-tuned GPT-2")
    
    print("\nIf both models give similar wrong answers, the fine-tuning failed.")
    print("The fine-tuned model should give correct TCS financial data.")