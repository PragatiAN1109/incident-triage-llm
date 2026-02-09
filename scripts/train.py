#!/usr/bin/env python3
"""
Fine-tuning script for incident triage model.
Uses Hugging Face Transformers Trainer API for clean, reproducible training.
"""

import json
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)


def load_jsonl_dataset(file_path: str) -> Dataset:
    """
    Load JSONL file into a Hugging Face Dataset.
    
    Args:
        file_path: Path to JSONL file with 'prompt' and 'response' fields
    
    Returns:
        Hugging Face Dataset object
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    return Dataset.from_list(data)


def preprocess_function(examples, tokenizer, max_input_length=512, max_target_length=256):
    """
    Tokenize inputs and targets for seq2seq training.
    
    Args:
        examples: Batch of examples with 'prompt' and 'response' fields
        tokenizer: Tokenizer to use
        max_input_length: Maximum input sequence length
        max_target_length: Maximum target sequence length
    
    Returns:
        Tokenized inputs and labels
    """
    # Tokenize inputs (prompts)
    model_inputs = tokenizer(
        examples["prompt"],
        max_length=max_input_length,
        truncation=True,
        padding=False  # Padding handled by data collator
    )
    
    # Tokenize targets (responses)
    labels = tokenizer(
        examples["response"],
        max_length=max_target_length,
        truncation=True,
        padding=False
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def create_training_args(output_dir: str):
    """
    Create Seq2SeqTrainingArguments compatible with different transformers versions.
    Handles both 'eval_strategy' (newer) and 'evaluation_strategy' (older) parameter names.
    """
    # Build arguments dict with newer parameter names (eval_strategy)
    args_dict = {
        "output_dir": output_dir,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "seed": 42,
        "report_to": "none",
        "predict_with_generate": False,
        "push_to_hub": False
    }
    
    # Try with newer parameter names first
    try:
        return Seq2SeqTrainingArguments(**args_dict)
    except TypeError as e:
        error_msg = str(e)
        
        # Handle eval_strategy → evaluation_strategy
        if "eval_strategy" in error_msg or "evaluation_strategy" in error_msg:
            args_dict["evaluation_strategy"] = args_dict.pop("eval_strategy")
            print("  Note: Using 'evaluation_strategy' (older transformers version)")
            return Seq2SeqTrainingArguments(**args_dict)
        else:
            # Re-raise if it's a different error
            raise


def main():
    """
    Main training pipeline.
    """
    print("="*70)
    print("INCIDENT TRIAGE MODEL - FINE-TUNING")
    print("="*70)
    
    # Configuration
    model_name = "google/flan-t5-small"
    train_file = "data/final/train.jsonl"
    val_file = "data/final/val.jsonl"
    output_dir = "results"
    final_model_dir = "results/final-model"
    
    # Verify files exist
    if not Path(train_file).exists():
        print(f"Error: Training file not found: {train_file}")
        return 1
    if not Path(val_file).exists():
        print(f"Error: Validation file not found: {val_file}")
        return 1
    
    # Load tokenizer and model
    print(f"\nLoading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print(f"✓ Model loaded successfully")
    
    # Load datasets
    print(f"\nLoading datasets...")
    print(f"  Train: {train_file}")
    train_dataset = load_jsonl_dataset(train_file)
    print(f"    Loaded {len(train_dataset)} training examples")
    
    print(f"  Validation: {val_file}")
    val_dataset = load_jsonl_dataset(val_file)
    print(f"    Loaded {len(val_dataset)} validation examples")
    
    # Tokenize datasets
    print(f"\nTokenizing datasets...")
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    print(f"✓ Tokenization complete")
    
    # Data collator for dynamic padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments (with version compatibility)
    print(f"\nConfiguring training parameters...")
    training_args = create_training_args(output_dir)
    
    print(f"  Output directory: {output_dir}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Seed: {training_args.seed}")
    
    # Initialize trainer
    print(f"\nInitializing Trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    print(f"✓ Trainer initialized")
    
    # Train
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}")
    
    train_result = trainer.train()
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    
    # Print final metrics
    metrics = train_result.metrics
    print(f"\nFinal Training Metrics:")
    print(f"  Training loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Training runtime: {metrics.get('train_runtime', 0):.2f}s")
    print(f"  Samples per second: {metrics.get('train_samples_per_second', 0):.2f}")
    
    # Evaluate on validation set
    print(f"\nEvaluating on validation set...")
    eval_metrics = trainer.evaluate()
    print(f"  Validation loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    
    # Save final model
    print(f"\nSaving final model to: {final_model_dir}")
    Path(final_model_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"✓ Model saved successfully")
    
    print(f"\n{'='*70}")
    print("✓ FINE-TUNING PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"\nFinal model saved to: {final_model_dir}")
    print(f"Use this model for inference on test data or new incidents.")
    
    return 0


if __name__ == "__main__":
    exit(main())
