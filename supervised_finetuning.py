import torch
import json
import time
import platform
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

def log_hyperparameters_and_setup():
    """Log all hyperparameters and compute setup"""
    
    # Improved Hyperparameters for better learning
    hyperparameters = {
        "model_name": "gpt2",
        "data_file": "finetuning_datasets/finetuning_data.json",
        "output_dir": "./sft_model",
        "per_device_train_batch_size": 4,  # Smaller batch for better convergence
        "gradient_accumulation_steps": 4,  # Effective batch size = 4*4 = 16
        "learning_rate": 1e-5,  # Lower learning rate for precise learning
        "num_train_epochs": 8,  # More epochs to learn the data better
        "warmup_steps": 50,  # Gradual learning rate warmup
        "weight_decay": 0.01,  # Regularization
        "logging_steps": 20,  # More frequent logging
        "save_steps": 100,  # Save checkpoints more frequently
        "logging_dir": "./logs",
        "max_length": 512,
        "temperature": 0.7,
        "final_model_dir": "./fine_tuned_sft_model_v2"  # New version
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
    # Preprocess function - simplified approach
    def preprocess_function(examples):
        # Combine instruction and output for causal language modeling
        texts = []
        for i in range(len(examples['instruction'])):
            text = f"{examples['instruction'][i]}\n{examples['output'][i]}{tokenizer.eos_token}"
            texts.append(text)
        
        # Tokenize the combined texts
        result = tokenizer(
            texts, 
            truncation=True, 
            padding=True,  # Enable padding to avoid tensor issues
            max_length=hyperparameters["max_length"],
            return_tensors=None  # Return as lists, not tensors
        )
        
        # For causal LM, labels are the same as input_ids
        result["labels"] = result["input_ids"]
        
        return result
    
    # Apply preprocessing and remove original columns
    tokenized_datasets = dataset.map(
        preprocess_function, 
        batched=True,
        remove_columns=dataset["train"].column_names  # Remove original columns
    )
    print("Dataset preprocessing completed")
    
    print("\n4. Defining Training Arguments...")
    # Improved training arguments for better convergence (compatible version)
    training_args = TrainingArguments(
        output_dir=hyperparameters["output_dir"],
        overwrite_output_dir=True,
        per_device_train_batch_size=hyperparameters["per_device_train_batch_size"],
        gradient_accumulation_steps=hyperparameters["gradient_accumulation_steps"],
        learning_rate=hyperparameters["learning_rate"],
        num_train_epochs=hyperparameters["num_train_epochs"],
        warmup_steps=hyperparameters["warmup_steps"],
        weight_decay=hyperparameters["weight_decay"],
        logging_dir=hyperparameters["logging_dir"],
        logging_steps=hyperparameters["logging_steps"],
        save_steps=hyperparameters["save_steps"],
        save_strategy="steps",
        remove_unused_columns=False,
        dataloader_drop_last=True,
        report_to="none",
        # Core optimization settings (most compatible)
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
    )
    
    print("Training arguments configured")
    
    print("\n5. Creating Trainer and Starting Fine-tuning...")
    # Use default data collator since we're padding in preprocessing
    from transformers import default_data_collator
    
    data_collator = default_data_collator
    
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
        train_result = trainer.train()
        training_time = time.time() - start_time
        
        print(f"\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        print(f"Final training loss: {train_result.training_loss:.6f}")
        print(f"Training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        print(f"Total training steps: {train_result.global_step}")
        print(f"Training samples processed: {train_result.global_step * hyperparameters['per_device_train_batch_size'] * hyperparameters['gradient_accumulation_steps']}")
        
        # Save model - following doc structure  
        print(f"\nSaving improved model to: {hyperparameters['final_model_dir']}")
        model.save_pretrained(hyperparameters["final_model_dir"])
        tokenizer.save_pretrained(hyperparameters["final_model_dir"])
        print("SFT Fine-tuning Done!")
        
        # Enhanced training log
        training_log = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "technique": "Supervised Instruction Fine-Tuning (SFT) - Improved",
            "version": "v2",
            "hyperparameters": hyperparameters,
            "compute_setup": compute_setup,
            "training_results": {
                "training_time_seconds": training_time,
                "final_train_loss": train_result.training_loss,
                "total_steps": train_result.global_step,
                "dataset_size": len(dataset['train']),
                "effective_batch_size": hyperparameters['per_device_train_batch_size'] * hyperparameters['gradient_accumulation_steps'],
                "total_samples_processed": train_result.global_step * hyperparameters['per_device_train_batch_size'] * hyperparameters['gradient_accumulation_steps']
            },
            "improvements": [
                "Lower learning rate (1e-5)",
                "More epochs (8)",
                "Gradient accumulation",
                "Cosine learning rate schedule",
                "Better regularization"
            ],
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