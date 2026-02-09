#!/usr/bin/env python3
"""
Final evaluation and inference demonstration script.
Loads the fine-tuned model and runs inference on held-out test samples.
"""

import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_model(model_path: str):
    """
    Load fine-tuned model and tokenizer.
    
    Args:
        model_path: Path to the fine-tuned model directory
    
    Returns:
        Tuple of (tokenizer, model)
    """
    print(f"Loading fine-tuned model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    print(f"✓ Model loaded successfully")
    return tokenizer, model


def generate_response(prompt: str, tokenizer, model, max_new_tokens: int = 256) -> str:
    """
    Generate triage response for a given incident prompt.
    
    Args:
        prompt: Input incident text
        tokenizer: Loaded tokenizer
        model: Loaded model
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Generated response text
    """
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        early_stopping=True,
        do_sample=False  # Deterministic generation
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def load_test_data(test_file: str):
    """
    Load test dataset.
    
    Args:
        test_file: Path to test JSONL file
    
    Returns:
        List of test examples
    """
    test_data = []
    with open(test_file, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    return test_data


def main():
    """
    Run inference on test set and display results.
    """
    print("="*70)
    print("FINAL EVALUATION & INFERENCE DEMO")
    print("="*70)
    
    # Configuration
    model_path = "results/config_c_higher_capacity"
    test_file = "data/final/test.jsonl"
    num_samples = 3  # Number of samples to display
    
    # Verify paths exist
    if not Path(model_path).exists():
        print(f"\nError: Model not found at {model_path}")
        print("Please run training experiments first: python3 scripts/train_experiments.py")
        return 1
    
    if not Path(test_file).exists():
        print(f"\nError: Test file not found at {test_file}")
        print("Please run build_dataset.py first to generate test data")
        return 1
    
    # Load model
    tokenizer, model = load_model(model_path)
    
    # Load test data
    print(f"\nLoading test data from: {test_file}")
    test_data = load_test_data(test_file)
    print(f"✓ Loaded {len(test_data)} test samples")
    
    # Run inference on samples
    print(f"\n{'='*70}")
    print(f"RUNNING INFERENCE ON {min(num_samples, len(test_data))} TEST SAMPLES")
    print(f"{'='*70}")
    
    for i, example in enumerate(test_data[:num_samples], 1):
        prompt = example["prompt"]
        ground_truth = example["response"]
        
        # Generate model output
        generated = generate_response(prompt, tokenizer, model)
        
        # Display results
        print(f"\n{'='*70}")
        print(f"TEST SAMPLE {i}")
        print(f"{'='*70}")
        
        print(f"\nINPUT:")
        print(f"{'-'*70}")
        # Show first 6 lines of prompt
        prompt_lines = prompt.split('\n')
        for line in prompt_lines[:6]:
            print(line)
        if len(prompt_lines) > 6:
            print(f"... ({len(prompt_lines) - 6} more lines)")
        
        print(f"\nMODEL OUTPUT:")
        print(f"{'-'*70}")
        print(generated)
        
        print(f"\nGROUND TRUTH:")
        print(f"{'-'*70}")
        # Try to pretty-print JSON, but don't fail if it's not valid JSON
        try:
            gt_data = json.loads(ground_truth)
            print(json.dumps(gt_data, indent=2))
        except:
            print(ground_truth)
    
    print(f"\n{'='*70}")
    print("INFERENCE DEMO COMPLETE")
    print(f"{'='*70}")
    print(f"\nModel: {model_path}")
    print(f"Test samples evaluated: {min(num_samples, len(test_data))} of {len(test_data)}")
    print(f"\nThis demonstrates the fine-tuned model's ability to:")
    print(f"  - Parse incident log patterns")
    print(f"  - Classify severity levels")
    print(f"  - Identify likely causes")
    print(f"  - Generate actionable recommendations")
    
    return 0


if __name__ == "__main__":
    exit(main())
