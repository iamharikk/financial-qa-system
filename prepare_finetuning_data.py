import json
import re
from pathlib import Path

def parse_qa_pairs(file_path):
    """
    Parse Q&A pairs from the text file and convert to fine-tuning format
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    qa_pairs = []
    
    # Use regex to find Question: ... Answer: ... patterns
    pattern = r'Question: (.+?) Answer: (.+?)(?=Question:|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for question, answer in matches:
        question = question.strip()
        answer = answer.strip()
        
        if question and answer:
            qa_pairs.append({
                'instruction': question,
                'output': answer
            })
    
    return qa_pairs

def create_finetuning_dataset(qa_pairs, output_path):
    """
    Create fine-tuning dataset in JSON format
    """
    # Convert to the format expected for fine-tuning
    finetuning_data = []
    
    for pair in qa_pairs:
        finetuning_data.append({
            'instruction': pair['instruction'],
            'output': pair['output']
        })
    
    # Save as JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(finetuning_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created fine-tuning dataset with {len(finetuning_data)} examples")
    print(f"Saved to: {output_path}")
    
    return finetuning_data

def main():
    # Paths
    qa_file = "data/q-and-a/qa-pairs.txt"
    output_file = "finetuning_datasets/finetuning_data.json"
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Parse Q&A pairs
    print("Parsing Q&A pairs...")
    qa_pairs = parse_qa_pairs(qa_file)
    print(f"Found {len(qa_pairs)} Q&A pairs")
    
    # Create fine-tuning dataset
    print("Creating fine-tuning dataset...")
    finetuning_data = create_finetuning_dataset(qa_pairs, output_file)
    
    # Show first few examples
    print("\nFirst 3 examples:")
    for i, example in enumerate(finetuning_data[:3]):
        print(f"\nExample {i+1}:")
        print(f"Instruction: {example['instruction']}")
        print(f"Output: {example['output']}")

if __name__ == "__main__":
    main()