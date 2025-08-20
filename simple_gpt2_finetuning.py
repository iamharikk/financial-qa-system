"""
Simple GPT-2 Fine-tuning Script
Easy-to-understand code for beginners to fine-tune GPT-2 on TCS Q/A data
"""

import json
import os
import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TCSQADataset(Dataset):
    """Simple dataset class for TCS Q/A data"""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 512):
        """
        Initialize dataset
        
        Args:
            data_file: Path to JSONL file with training data
            tokenizer: GPT-2 tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load data
        print(f"Loading data from {data_file}...")
        self.load_data(data_file)
        print(f"Loaded {len(self.examples)} examples")
    
    def load_data(self, data_file: str):
        """Load data from JSONL file"""
        if not os.path.exists(data_file):
            print(f"Error: File {data_file} not found!")
            return
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    
                    # Handle different formats
                    if 'prompt' in item and 'completion' in item:
                        # Prompt-completion format
                        text = item['prompt'] + item['completion']
                    elif 'messages' in item:
                        # Conversation format - combine user and assistant messages only
                        text = ""
                        for message in item['messages']:
                            if message['role'] == 'user':
                                text += f"Question: {message['content']} "
                            elif message['role'] == 'assistant':
                                text += f"Answer: {message['content']}"
                        # Skip if no user/assistant messages found
                        if not text.strip():
                            continue
                    else:
                        continue
                    
                    # Add special tokens
                    text = text + self.tokenizer.eos_token
                    self.examples.append(text)
                    
                except Exception as e:
                    print(f"Warning: Could not parse line: {e}")
                    print(f"Problem line content: {line[:100]}...")
                    continue
    
    def __len__(self):
        """Return dataset size"""
        return len(self.examples)
    
    def __getitem__(self, idx):
        """Get one example"""
        text = self.examples[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()  # For language modeling
        }

class SimpleGPT2FineTuner:
    """Simple GPT-2 fine-tuning class for beginners"""
    
    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize fine-tuner
        
        Args:
            model_name: Model to fine-tune (default: gpt2)
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        print(f"Setting up fine-tuning for {model_name}")
        self.setup_model()
    
    def setup_model(self):
        """Load model and tokenizer"""
        print("Loading tokenizer and model...")
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        
        # Add padding token (GPT-2 doesn't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        
        # Resize embeddings if we added tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        print("Model and tokenizer loaded successfully")
    
    def prepare_datasets(self, train_file: str, val_file: str = None):
        """
        Prepare training and validation datasets
        
        Args:
            train_file: Path to training JSONL file
            val_file: Path to validation JSONL file (optional)
        """
        print("Preparing datasets...")
        
        # Load training data
        self.train_dataset = TCSQADataset(train_file, self.tokenizer)
        
        # Load validation data if provided
        if val_file and os.path.exists(val_file):
            self.val_dataset = TCSQADataset(val_file, self.tokenizer)
        else:
            print("No validation file provided")
            self.val_dataset = None
        
        print("Datasets prepared")
    
    def setup_training(self, output_dir: str = "fine_tuned_gpt2", 
                      num_epochs: int = 3, learning_rate: float = 5e-5):
        """
        Setup training configuration
        
        Args:
            output_dir: Directory to save the fine-tuned model
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training
        """
        print("Setting up training configuration...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=2,  # Small batch size for limited memory
            per_device_eval_batch_size=2,
            warmup_steps=100,
            learning_rate=learning_rate,
            logging_steps=50,
            logging_dir=f"{output_dir}/logs",
            save_strategy="epoch",
            eval_strategy="epoch" if self.val_dataset else "no",  # Changed from evaluation_strategy
            load_best_model_at_end=True if self.val_dataset else False,
            metric_for_best_model="eval_loss" if self.val_dataset else None,
            report_to=None,  # Don't report to wandb/tensorboard
            save_total_limit=2,  # Only keep 2 checkpoints
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # GPT-2 uses causal language modeling, not masked
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
        )
        
        print("Training setup complete")
    
    def train(self):
        """Start training"""
        if self.trainer is None:
            print("Error: Training not set up. Call setup_training() first")
            return
        
        print("Starting training...")
        print("This may take several minutes to hours depending on your hardware")
        
        try:
            # Train the model
            self.trainer.train()
            
            print("Training completed successfully!")
            
            # Save the final model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.trainer.args.output_dir)
            
            print(f"Model saved to: {self.trainer.args.output_dir}")
            
        except Exception as e:
            print(f"Training failed: {e}")
    
    def test_model(self, test_prompt: str = "Question: What is TCS revenue? Answer:"):
        """
        Test the fine-tuned model with a sample prompt
        
        Args:
            test_prompt: Prompt to test the model
        """
        print(f"Testing model with prompt: '{test_prompt}'")
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(test_prompt, return_tensors='pt')
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(test_prompt):].strip()
            
            print(f"Model response: {response}")
            
        except Exception as e:
            print(f"Testing failed: {e}")

def main():
    """Main function to run fine-tuning"""
    print("=" * 60)
    print("SIMPLE GPT-2 FINE-TUNING")
    print("=" * 60)
    
    # Check if datasets exist (try conversation format first, then prompt-completion)
    train_file = "finetuning_datasets/train_conversation.jsonl"
    val_file = "finetuning_datasets/val_conversation.jsonl"
    
    # If conversation format doesn't exist, try prompt-completion format
    if not os.path.exists(train_file):
        train_file = "finetuning_datasets/train_prompt_completion.jsonl"
        val_file = "finetuning_datasets/val_prompt_completion.jsonl"
    
    if not os.path.exists(train_file):
        print(f"Error: Training file {train_file} not found!")
        print("Please run prepare_finetuning_dataset.py first")
        return
    
    # Create fine-tuner
    finetuner = SimpleGPT2FineTuner("gpt2")
    
    # Prepare datasets
    finetuner.prepare_datasets(train_file, val_file)
    
    # Setup training (you can modify these parameters)
    finetuner.setup_training(
        output_dir="fine_tuned_gpt2_tcs",
        num_epochs=3,           # Start with 3 epochs
        learning_rate=5e-5      # Standard learning rate
    )
    
    # Train the model
    finetuner.train()
    
    # Test the model
    print("\n" + "=" * 40)
    print("TESTING FINE-TUNED MODEL")
    print("=" * 40)
    
    test_questions = [
        "Question: What is TCS revenue? Answer:",
        "Question: What is TCS profit? Answer:",
        "Question: How much cash does TCS have? Answer:"
    ]
    
    for question in test_questions:
        print(f"\n{question}")
        finetuner.test_model(question)
    
    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE!")
    print("=" * 60)
    print("Your fine-tuned model is saved in: fine_tuned_gpt2_tcs/")
    print("You can now use this model for TCS financial Q/A!")

if __name__ == "__main__":
    main()