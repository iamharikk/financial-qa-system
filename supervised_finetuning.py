import torch
import json
import time
import platform
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

def log_hyperparameters_and_setup():
    """Log all hyperparameters and compute setup"""
    
    # Hyperparameters
    hyperparameters = {
        "model_name": "gpt2",
        "data_file": "finetuning_datasets/finetuning_data.json",
        "output_dir": "./sft_model",
        "per_device_train_batch_size": 8,
        "learning_rate": 2e-5,
        "num_train_epochs": 3,
        "logging_dir": "./logs",
        "max_length": 512,
        "temperature": 0.7,
        "final_model_dir": "./fine_tuned_sft_model"
    }
    
    # Compute setup
    compute_setup = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cpu_count": torch.get_num_threads(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_available": True
    }
    
    print("=" * 80)
    print("SUPERVISED FINE-TUNING (SFT) SETUP")
    print("=" * 80)
    
    print("\nHYPERPARAMETERS:")
    print("-" * 50)
    for key, value in hyperparameters.items():
        print(f"{key}: {value}")
    
    print("\nCOMPUTE SETUP:")
    print("-" * 50)
    for key, value in compute_setup.items():
        print(f"{key}: {value}")
    
    print("=" * 80)
    
    return hyperparameters, compute_setup

def main():
    """Main SFT function following the documentation structure"""
    
    # Log setup
    hyperparameters, compute_setup = log_hyperparameters_and_setup()
    
    print("\n1. Loading Supervised Fine-tuning Dataset...")
    # Load dataset - following the doc structure
    dataset = load_dataset("json", data_files=hyperparameters["data_file"])
    print(f"Dataset loaded with {len(dataset['train'])} examples")
    
    # Show dataset structure
    print(f"First example: {dataset['train'][0]}")
    
    print("\n2. Loading Pre-trained Language Model and Tokenizer...")
    # Load model and tokenizer - exactly as in doc
    model_name = hyperparameters["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if not present (from doc)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    print(f"Model: {model_name}")
    print(f"Model parameters: {model.num_parameters():,}")
    
    print("\n3. Preprocessing the Dataset...")
    # Preprocess function - following doc structure exactly
    def preprocess_function(examples):
        # Combine instruction and output for causal language modeling
        texts = []
        for i in range(len(examples['instruction'])):
            text = f"{examples['instruction'][i]}\n{examples['output'][i]}{tokenizer.eos_token}"
            texts.append(text)
        
        # Tokenize the combined texts
        model_inputs = tokenizer(
            texts, 
            truncation=True, 
            padding=False,  # Don't pad here, let data collator handle it
            max_length=hyperparameters["max_length"]
        )
        
        # For causal LM, labels are the same as input_ids
        model_inputs["labels"] = [ids[:] for ids in model_inputs["input_ids"]]
        
        return model_inputs
    
    # Apply preprocessing and remove original columns
    tokenized_datasets = dataset.map(
        preprocess_function, 
        batched=True,
        remove_columns=dataset["train"].column_names  # Remove original columns
    )
    print("Dataset preprocessing completed")
    
    print("\n4. Defining Training Arguments...")
    # Training arguments - following doc structure
    training_args = TrainingArguments(
        output_dir=hyperparameters["output_dir"],
        per_device_train_batch_size=hyperparameters["per_device_train_batch_size"],
        learning_rate=hyperparameters["learning_rate"],
        num_train_epochs=hyperparameters["num_train_epochs"],
        logging_dir=hyperparameters["logging_dir"],
        remove_unused_columns=False,  # Keep 'instruction' and 'output' for potential later use
        report_to="none"  # Or "tensorboard" if you have it installed
    )
    
    print("Training arguments configured")
    
    print("\n5. Creating Trainer and Starting Fine-tuning...")
    # Use proper data collator for language modeling
    from transformers import DataCollatorForLanguageModeling
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Create trainer - exactly as in doc
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )
    
    # Start training with timing
    print("Starting fine-tuning...")
    start_time = time.time()
    
    try:
        trainer.train()
        training_time = time.time() - start_time
        
        print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        
        # Save model - following doc structure
        model.save_pretrained(hyperparameters["final_model_dir"])
        tokenizer.save_pretrained(hyperparameters["final_model_dir"])
        print("SFT Fine-tuning Done!")
        
        # Save training log
        training_log = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "technique": "Supervised Instruction Fine-Tuning (SFT)",
            "hyperparameters": hyperparameters,
            "compute_setup": compute_setup,
            "training_time_seconds": training_time,
            "dataset_size": len(dataset['train']),
            "status": "completed"
        }
        
        with open('sft_training_log.json', 'w') as f:
            json.dump(training_log, f, indent=2)
        
        print(f"Training log saved to: sft_training_log.json")
        
        # Test the model
        print("\n6. Testing the Fine-tuned Model...")
        test_question = "What was TCS's net profit in Mar '25?"
        
        # Format input as in preprocessing
        input_text = f"{test_question}\n{tokenizer.bos_token}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=input_ids.shape[1] + 50,
                num_return_sequences=1,
                temperature=hyperparameters["temperature"],
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode and print response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_answer = response[len(test_question)+1:].strip()
        
        print(f"Test Question: {test_question}")
        print(f"Generated Answer: {generated_answer}")
        
        print("\n" + "=" * 80)
        print("SUPERVISED FINE-TUNING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("Fine-tuning failed!")
        exit(1)