#!/usr/bin/env python3
"""
Model setup and baseline inference script.
Loads the base FLAN-T5-small model and demonstrates inference before fine-tuning.
"""

import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_model(model_name: str = "google/flan-t5-small"):
    """
    Load pre-trained model and tokenizer from Hugging Face.
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print(f"✓ Model loaded successfully")
    return tokenizer, model


def generate_response(
    prompt: str,
    tokenizer,
    model,
    max_new_tokens: int = 256,
    num_beams: int = 4,
    temperature: float = 0.7
) -> str:
    """
    Generate a response from the model given a prompt.
    
    Args:
        prompt: Input text (incident description)
        tokenizer: Loaded tokenizer
        model: Loaded model
        max_new_tokens: Maximum tokens to generate
        num_beams: Number of beams for beam search
        temperature: Sampling temperature
    
    Returns:
        Generated text response
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate output
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        temperature=temperature,
        do_sample=False,  # Use greedy/beam search for deterministic output
        early_stopping=True
    )
    
    # Decode to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def main():
    """
    Demonstrate baseline model inference on a sample from the training set.
    """
    print("="*70)
    print("BASELINE MODEL INFERENCE (Before Fine-Tuning)")
    print("="*70)
    
    # Load model
    tokenizer, model = load_model("google/flan-t5-small")
    
    # Load one example from training data
    train_file = Path("data/final/train.jsonl")
    
    if not train_file.exists():
        print(f"\nError: Training file not found at {train_file}")
        print("Please run build_dataset.py first to generate training data.")
        return 1
    
    print(f"\nLoading sample from: {train_file}")
    with open(train_file, 'r') as f:
        first_line = f.readline()
        example = json.loads(first_line)
    
    prompt = example["prompt"]
    ground_truth = example["response"]
    
    # Show prompt (truncated for display)
    print(f"\n{'='*70}")
    print("INPUT PROMPT (truncated):")
    print(f"{'='*70}")
    prompt_lines = prompt.split('\n')
    for line in prompt_lines[:6]:  # Show first 6 lines
        print(line)
    if len(prompt_lines) > 6:
        print(f"... ({len(prompt_lines) - 6} more lines)")
    
    # Generate baseline response
    print(f"\n{'='*70}")
    print("BASELINE MODEL OUTPUT (without fine-tuning):")
    print(f"{'='*70}")
    generated = generate_response(prompt, tokenizer, model)
    print(generated)
    
    # Show ground truth
    print(f"\n{'='*70}")
    print("GROUND TRUTH RESPONSE:")
    print(f"{'='*70}")
    # Parse and pretty-print JSON
    try:
        gt_data = json.loads(ground_truth)
        print(json.dumps(gt_data, indent=2))
    except:
        print(ground_truth)
    
    print(f"\n{'='*70}")
    print("NOTES:")
    print(f"{'='*70}")
    print("The baseline model has NOT been fine-tuned on incident triage.")
    print("Output quality will improve significantly after fine-tuning on our dataset.")
    print("Expected improvements:")
    print("  - Structured JSON format matching ground truth")
    print("  - Accurate severity classification")
    print("  - Domain-specific cause identification")
    print("  - Actionable remediation recommendations")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    exit(main())
