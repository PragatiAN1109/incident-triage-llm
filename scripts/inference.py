#!/usr/bin/env python3
"""
Final evaluation and inference demonstration script.
Loads the fine-tuned model and runs inference on held-out test samples.
Automatically detects and loads the latest checkpoint if needed.
"""

import json
import argparse
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def find_model_path(base_path: Path) -> Path:
    """
    Find the model path, checking for final-model first, then latest checkpoint.
    
    Args:
        base_path: Base directory to search
    
    Returns:
        Path to the model
    """
    # Priority 1: Check for final-model subdirectory
    final_model_path = base_path / "final-model"
    if final_model_path.exists() and (final_model_path / "config.json").exists():
        print(f"Loading final model from: {final_model_path}")
        return final_model_path
    
    # Priority 2: Check if base_path itself has config.json
    if (base_path / "config.json").exists():
        return base_path
    
    # Priority 3: Look for checkpoint subdirectories
    checkpoint_pattern = re.compile(r'checkpoint-(\d+)')
    checkpoints = []
    
    for item in base_path.iterdir():
        if item.is_dir():
            match = checkpoint_pattern.match(item.name)
            if match:
                step_num = int(match.group(1))
                checkpoints.append((step_num, item))
    
    if checkpoints:
        # Sort by step number and get the latest
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        latest_step, latest_path = checkpoints[0]
        print(f"No final-model found at {base_path}")
        print(f"Loading latest checkpoint: {latest_path.name}")
        return latest_path
    
    # No valid model found
    return base_path


def load_model(model_path: str):
    """
    Load fine-tuned model and tokenizer.
    Automatically detects final-model or latest checkpoint.
    
    Args:
        model_path: Path to the fine-tuned model directory
    
    Returns:
        Tuple of (tokenizer, model)
    """
    base_path = Path(model_path)
    
    if not base_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Find the actual model path
    resolved_path = find_model_path(base_path)
    
    print(f"Loading model from: {resolved_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(resolved_path))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(resolved_path))
    print(f"✓ Model loaded successfully")
    return tokenizer, model


def wrap_prompt_with_instruction(prompt: str) -> str:
    """
    Wrap the incident prompt with instruction context to guide generation.
    
    Args:
        prompt: Raw incident text
    
    Returns:
        Wrapped prompt with instructions
    """
    instruction = (
        "You are an incident triage assistant. Return ONLY valid JSON with keys: "
        "severity, likely_cause, recommended_action.\n\n"
        f"INCIDENT:\n{prompt}\n\nJSON:"
    )
    return instruction


def check_format(generated_text: str) -> str:
    """
    Check if generated text contains valid JSON format.
    
    Args:
        generated_text: Model output
    
    Returns:
        Format check status message
    """
    try:
        # Try to parse as JSON
        data = json.loads(generated_text)
        
        # Check required keys
        required_keys = ["severity", "likely_cause", "recommended_action"]
        has_all_keys = all(key in data for key in required_keys)
        
        # Check severity format (SEV-1, SEV-2, or SEV-3)
        severity_valid = False
        if "severity" in data:
            severity_valid = re.match(r'^SEV-[123]$', str(data["severity"])) is not None
        
        if has_all_keys and severity_valid:
            return "✓ FORMAT OK"
        elif has_all_keys:
            return "⚠ FORMAT WARNING: Invalid severity format"
        else:
            return "⚠ FORMAT WARNING: Missing required keys"
    except:
        return "⚠ FORMAT WARNING: Not valid JSON"


def generate_response(prompt: str, tokenizer, model, max_new_tokens: int = 128) -> str:
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
    # Wrap prompt with instruction
    instructed_prompt = wrap_prompt_with_instruction(prompt)
    
    inputs = tokenizer(instructed_prompt, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        early_stopping=True,
        do_sample=False,  # Deterministic generation
        repetition_penalty=1.2,  # Penalize repetition
        no_repeat_ngram_size=3  # Prevent repeating 3-grams
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
    parser = argparse.ArgumentParser(
        description="Run inference on test data using fine-tuned model"
    )
    parser.add_argument(
        "--model_path",
        default="results/config_c_(higher_capacity)",
        help="Path to fine-tuned model or experiment directory"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of test samples to display"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("FINAL EVALUATION & INFERENCE DEMO")
    print("="*70)
    
    # Configuration
    model_path = args.model_path
    test_file = "data/final/test.jsonl"
    num_samples = args.num_samples
    
    # Verify paths exist
    if not Path(model_path).exists():
        print(f"\nError: Model path not found: {model_path}")
        print("Please run training first: python3 scripts/train.py")
        return 1
    
    if not Path(test_file).exists():
        print(f"\nError: Test file not found: {test_file}")
        print("Please run build_dataset.py first to generate test data")
        return 1
    
    # Load model (with automatic checkpoint detection)
    try:
        tokenizer, model = load_model(model_path)
    except Exception as e:
        print(f"\nError loading model: {e}")
        return 1
    
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
        
        # Check format
        format_status = check_format(generated)
        
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
        print(f"\n{format_status}")
        
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
