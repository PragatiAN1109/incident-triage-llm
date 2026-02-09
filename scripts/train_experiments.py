#!/usr/bin/env python3
"""
Hyperparameter optimization experiment runner.
Trains multiple configurations and compares validation performance.
Supports --only_best flag to run only Config C (best configuration).
"""

import json
import time
import argparse
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
    """Load JSONL file into a Hugging Face Dataset."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


def preprocess_function(examples, tokenizer, max_input_length=512, max_target_length=256):
    """Tokenize inputs and targets for seq2seq training."""
    model_inputs = tokenizer(
        examples["prompt"],
        max_length=max_input_length,
        truncation=True,
        padding=False
    )
    
    labels = tokenizer(
        examples["response"],
        max_length=max_target_length,
        truncation=True,
        padding=False
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def create_training_args(output_dir: str, learning_rate: float, batch_size: int, epochs: int):
    """
    Create Seq2SeqTrainingArguments compatible with different transformers versions.
    """
    args_dict = {
        "output_dir": output_dir,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
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
    
    try:
        return Seq2SeqTrainingArguments(**args_dict)
    except TypeError as e:
        if "eval_strategy" in str(e) or "evaluation_strategy" in str(e):
            args_dict["evaluation_strategy"] = args_dict.pop("eval_strategy")
            return Seq2SeqTrainingArguments(**args_dict)
        else:
            raise


def run_experiment(
    config_name: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    train_dataset,
    val_dataset,
    tokenizer,
    save_final_model: bool = True
):
    """
    Run a single training experiment with given hyperparameters.
    
    Args:
        save_final_model: If True, saves model+tokenizer to final-model subdirectory after training
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {config_name}")
    print(f"{'='*70}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    
    output_dir = f"results/{config_name.lower().replace(' ', '_')}"
    
    # Load fresh model for each experiment
    print(f"\nLoading fresh model...")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
    training_args = create_training_args(output_dir, learning_rate, batch_size, epochs)
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Train
    print(f"\nStarting training...")
    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()
    
    # Evaluate
    eval_metrics = trainer.evaluate()
    
    # Save final model if requested (for best config)
    if save_final_model:
        final_model_dir = Path(output_dir) / "final-model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving final model to: {final_model_dir}")
        model.save_pretrained(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))
        
        # Verify config.json exists
        config_path = final_model_dir / "config.json"
        if config_path.exists():
            print(f"✓ Verified config.json exists at: {config_path}")
        else:
            print(f"⚠ Warning: config.json not found at: {config_path}")
    
    # Collect results
    results = {
        "config_name": config_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "train_loss": train_result.metrics.get("train_loss", None),
        "val_loss": eval_metrics.get("eval_loss", None),
        "runtime_seconds": end_time - start_time,
        "output_dir": output_dir
    }
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {config_name}")
    print(f"{'='*70}")
    print(f"  Training loss: {results['train_loss']:.4f}")
    print(f"  Validation loss: {results['val_loss']:.4f}")
    print(f"  Runtime: {results['runtime_seconds']:.2f}s ({results['runtime_seconds']/60:.2f} min)")
    print(f"  Model saved to: {output_dir}")
    if save_final_model:
        print(f"  Final model saved to: {output_dir}/final-model")
    
    return results


def main():
    """
    Run hyperparameter optimization experiments.
    """
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization experiments"
    )
    parser.add_argument(
        "--only_best",
        action="store_true",
        help="Run only Config C (best configuration) instead of all experiments"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    if args.only_best:
        print("TRAINING BEST CONFIGURATION (Config C)")
    else:
        print("HYPERPARAMETER OPTIMIZATION EXPERIMENTS")
    print("="*70)
    
    # Load tokenizer (shared across all experiments)
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    
    # Load and tokenize datasets (shared across all experiments)
    print(f"\nLoading datasets...")
    train_dataset = load_jsonl_dataset("data/final/train.jsonl")
    val_dataset = load_jsonl_dataset("data/final/val.jsonl")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    
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
    
    # Define experiment configurations
    configs = [
        {
            "name": "Config A (Baseline)",
            "learning_rate": 5e-5,
            "batch_size": 4,
            "epochs": 3
        },
        {
            "name": "Config B (Lower LR)",
            "learning_rate": 2e-5,
            "batch_size": 4,
            "epochs": 3
        },
        {
            "name": "Config C (Higher Capacity)",
            "learning_rate": 5e-5,
            "batch_size": 2,
            "epochs": 5
        }
    ]
    
    # Filter to only best config if requested
    if args.only_best:
        configs = [c for c in configs if c["name"] == "Config C (Higher Capacity)"]
        print(f"\n✓ Running only best configuration: Config C")
        print(f"  Learning rate: {configs[0]['learning_rate']}")
        print(f"  Batch size: {configs[0]['batch_size']}")
        print(f"  Epochs: {configs[0]['epochs']}")
    
    # Run experiments
    all_results = []
    for config in configs:
        # Save final model only for Config C
        save_final = (config["name"] == "Config C (Higher Capacity)")
        
        result = run_experiment(
            config["name"],
            config["learning_rate"],
            config["batch_size"],
            config["epochs"],
            tokenized_train,
            tokenized_val,
            tokenizer,
            save_final_model=save_final
        )
        all_results.append(result)
    
    # Summary comparison (only if running multiple configs)
    if not args.only_best and len(all_results) > 1:
        print(f"\n{'='*70}")
        print("EXPERIMENT COMPARISON")
        print(f"{'='*70}")
        print(f"\n{'Config':<25} {'Train Loss':<12} {'Val Loss':<12} {'Runtime (min)':<15}")
        print(f"{'-'*70}")
        
        for r in all_results:
            print(f"{r['config_name']:<25} {r['train_loss']:<12.4f} {r['val_loss']:<12.4f} {r['runtime_seconds']/60:<15.2f}")
        
        # Identify best configuration
        best_config = min(all_results, key=lambda x: x['val_loss'])
        print(f"\n{'='*70}")
        print(f"BEST CONFIGURATION: {best_config['config_name']}")
        print(f"{'='*70}")
        print(f"  Validation loss: {best_config['val_loss']:.4f}")
        print(f"  Training loss: {best_config['train_loss']:.4f}")
        print(f"  Runtime: {best_config['runtime_seconds']/60:.2f} minutes")
        print(f"  Model saved to: {best_config['output_dir']}")
    
    # Save results to JSON
    results_file = "results/experiment_results.json"
    Path("results").mkdir(exist_ok=True)
    
    if args.only_best:
        # For --only_best, just save the single result
        with open(results_file, 'w') as f:
            json.dump({
                "mode": "best_only",
                "experiment": all_results[0],
                "config": "Config C (Higher Capacity)"
            }, f, indent=2)
        print(f"\n✓ Results saved to: {results_file}")
    else:
        # For full experiments, save comparison
        best_config = min(all_results, key=lambda x: x['val_loss'])
        with open(results_file, 'w') as f:
            json.dump({
                "experiments": all_results,
                "best_config": best_config['config_name'],
                "best_val_loss": best_config['val_loss']
            }, f, indent=2)
        print(f"\n✓ Results saved to: {results_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
