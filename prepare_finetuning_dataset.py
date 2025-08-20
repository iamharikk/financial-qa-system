"""
Fine-tuning Dataset Preparation Script
Converts Q/A pairs into format suitable for fine-tuning language models
"""

import json
import os
from typing import List, Dict

class FineTuningDatasetPreparator:
    """Simple class to prepare fine-tuning dataset from Q/A pairs"""
    
    def __init__(self, qa_file_path: str):
        """
        Initialize with path to Q/A pairs file
        
        Args:
            qa_file_path: Path to the qa-pairs.txt file
        """
        self.qa_file_path = qa_file_path
        self.qa_pairs = []
        
    def load_qa_pairs(self) -> List[Dict[str, str]]:
        """
        Load Q/A pairs from the text file
        
        Returns:
            List of dictionaries with 'question' and 'answer' keys
        """
        print(f"Loading Q/A pairs from: {self.qa_file_path}")
        
        if not os.path.exists(self.qa_file_path):
            print(f"Error: File {self.qa_file_path} not found!")
            return []
        
        qa_pairs = []
        
        try:
            with open(self.qa_file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # Skip empty lines and headers
                if not line or "TCS Q&A Dataset" in line or "Financial Data" in line:
                    continue
                
                # Look for lines with "Question:" and "Answer:"
                if "Question:" in line and "Answer:" in line:
                    try:
                        # Split on "Answer:" to separate question and answer
                        parts = line.split("Answer:", 1)
                        
                        if len(parts) == 2:
                            question = parts[0].replace("Question:", "").strip()
                            answer = parts[1].strip()
                            
                            if question and answer:
                                qa_pairs.append({
                                    'question': question,
                                    'answer': answer,
                                    'line_number': line_num
                                })
                    except Exception as e:
                        print(f"Warning: Could not parse line {line_num}: {e}")
                        continue
            
            print(f"Successfully loaded {len(qa_pairs)} Q/A pairs")
            self.qa_pairs = qa_pairs
            return qa_pairs
            
        except Exception as e:
            print(f"Error loading file: {e}")
            return []
    
    def create_conversation_format(self) -> List[Dict]:
        """
        Convert Q/A pairs to conversation format for fine-tuning
        
        Format: Each entry has a "messages" list with system, user, and assistant messages
        """
        print("Converting to conversation format...")
        
        conversation_data = []
        
        system_message = "You are a helpful financial assistant specializing in TCS financial data. Provide accurate and concise answers based on the financial information you have been trained on."
        
        for i, qa_pair in enumerate(self.qa_pairs, 1):
            conversation = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user", 
                        "content": qa_pair['question']
                    },
                    {
                        "role": "assistant",
                        "content": qa_pair['answer']
                    }
                ],
                "id": f"tcs_qa_{i}"
            }
            
            conversation_data.append(conversation)
        
        print(f"Created {len(conversation_data)} conversation examples")
        return conversation_data
    
    def create_prompt_completion_format(self) -> List[Dict]:
        """
        Convert Q/A pairs to prompt-completion format (simpler format)
        
        Format: Each entry has "prompt" and "completion" fields
        """
        print("Converting to prompt-completion format...")
        
        prompt_completion_data = []
        
        for i, qa_pair in enumerate(self.qa_pairs, 1):
            # Create a formatted prompt
            prompt = f"You are a financial assistant. Answer the following question about TCS:\\n\\nQuestion: {qa_pair['question']}\\nAnswer:"
            
            # The completion is just the answer
            completion = f" {qa_pair['answer']}"
            
            entry = {
                "prompt": prompt,
                "completion": completion,
                "id": f"tcs_qa_{i}"
            }
            
            prompt_completion_data.append(entry)
        
        print(f"Created {len(prompt_completion_data)} prompt-completion examples")
        return prompt_completion_data
    
    def split_dataset(self, data: List[Dict], train_ratio: float = 0.8) -> tuple:
        """
        Split dataset into training and validation sets
        
        Args:
            data: List of training examples
            train_ratio: Fraction of data to use for training (0.8 = 80%)
        
        Returns:
            Tuple of (train_data, validation_data)
        """
        print(f"Splitting dataset with {train_ratio:.1%} for training...")
        
        total_examples = len(data)
        train_size = int(total_examples * train_ratio)
        
        train_data = data[:train_size]
        val_data = data[train_size:]
        
        print(f"Training examples: {len(train_data)}")
        print(f"Validation examples: {len(val_data)}")
        
        return train_data, val_data
    
    def save_dataset(self, data: List[Dict], filename: str, output_dir: str = "finetuning_datasets"):
        """
        Save dataset to JSON file
        
        Args:
            data: Dataset to save
            filename: Output filename
            output_dir: Output directory
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Save as JSONL (one JSON object per line) - common format for fine-tuning
                for item in data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\\n')
            
            print(f"Dataset saved to: {output_path}")
            print(f"Total examples: {len(data)}")
            
            # Show sample entry
            if data:
                print(f"\\nSample entry:")
                print(json.dumps(data[0], indent=2, ensure_ascii=False)[:300] + "...")
            
        except Exception as e:
            print(f"Error saving dataset: {e}")
    
    def create_all_formats(self, output_dir: str = "finetuning_datasets"):
        """
        Create datasets in both conversation and prompt-completion formats
        """
        print("=" * 60)
        print("FINE-TUNING DATASET PREPARATION")
        print("=" * 60)
        
        # Step 1: Load Q/A pairs
        qa_pairs = self.load_qa_pairs()
        if not qa_pairs:
            print("No Q/A pairs loaded. Stopping.")
            return
        
        print(f"\\nSample Q/A pair:")
        print(f"Q: {qa_pairs[0]['question']}")
        print(f"A: {qa_pairs[0]['answer']}")
        
        # Step 2: Create conversation format
        print(f"\\n" + "=" * 40)
        print("CONVERSATION FORMAT (for modern models)")
        print("=" * 40)
        conversation_data = self.create_conversation_format()
        
        # Split and save conversation format
        train_conv, val_conv = self.split_dataset(conversation_data)
        self.save_dataset(train_conv, "train_conversation.jsonl", output_dir)
        self.save_dataset(val_conv, "val_conversation.jsonl", output_dir)
        
        # Step 3: Create prompt-completion format
        print(f"\\n" + "=" * 40)
        print("PROMPT-COMPLETION FORMAT (for older models)")
        print("=" * 40)
        prompt_completion_data = self.create_prompt_completion_format()
        
        # Split and save prompt-completion format
        train_pc, val_pc = self.split_dataset(prompt_completion_data)
        self.save_dataset(train_pc, "train_prompt_completion.jsonl", output_dir)
        self.save_dataset(val_pc, "val_prompt_completion.jsonl", output_dir)
        
        # Step 4: Create summary
        print(f"\\n" + "=" * 60)
        print("DATASET PREPARATION COMPLETE!")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Total Q/A pairs processed: {len(qa_pairs)}")
        print(f"\\nFiles created:")
        print(f"   - train_conversation.jsonl ({len(train_conv)} examples)")
        print(f"   - val_conversation.jsonl ({len(val_conv)} examples)")
        print(f"   - train_prompt_completion.jsonl ({len(train_pc)} examples)")
        print(f"   - val_prompt_completion.jsonl ({len(val_pc)} examples)")
        
        print(f"\\nUsage:")
        print(f"   - Use conversation format for GPT-3.5/GPT-4 fine-tuning")
        print(f"   - Use prompt-completion format for older models or custom training")
        print(f"   - Training files contain 80% of data, validation files contain 20%")

def main():
    """Main function to run the dataset preparation"""
    
    # Path to your Q/A pairs file
    qa_file_path = "data/q-and-a/qa-pairs.txt"
    
    # Create dataset preparator
    preparator = FineTuningDatasetPreparator(qa_file_path)
    
    # Create all dataset formats
    preparator.create_all_formats()

if __name__ == "__main__":
    main()