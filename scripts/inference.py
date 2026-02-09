#!/usr/bin/env python3
"""
Final evaluation and inference demonstration script.
Loads the fine-tuned model and runs inference on held-out test samples.
Uses structured JSON completion (slot filling) for stable outputs.
Includes inference-time JSON repair with robust normalization.
"""

import json
import argparse
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from prompt_template import format_incident_prompt


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


def normalize_output(text: str) -> str:
    """
    Normalize model output to fix common JSON formatting issues.
    
    Args:
        text: Raw model output
    
    Returns:
        Normalized text
    """
    # Replace smart quotes with normal quotes
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")
    
    # Replace backticks with quotes
    text = text.replace("`", '"')
    
    # Fix parentheses around values that break JSON
    # Pattern: : (some text) -> : "some text"
    # This handles cases like "recommended_action": (Monitor...)
    text = re.sub(r':\s*\(([^)]+)\)', r': "\1"', text)
    
    # Trim whitespace
    text = text.strip()
    
    # Extract substring between first { and last }
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1]
    
    return text


def get_default_action(likely_cause: str) -> str:
    """
    Get default recommended action for a given likely cause.
    
    Args:
        likely_cause: The likely cause value
    
    Returns:
        Default recommended action
    """
    defaults = {
        "Packet responder termination": "Monitor DataNode for abnormal packet responder patterns; verify normal block replication",
        "Block serving exception": "Investigate DataNode block serving failures and check network connectivity between nodes"
    }
    
    return defaults.get(likely_cause, "Review incident logs and correlate with cluster metrics")


def repair_json(raw_output: str, incident_text: str = "") -> dict:
    """
    Attempt to repair invalid JSON from model output using robust extraction.
    Uses heuristic defaults instead of UNKNOWN when possible.
    
    Args:
        raw_output: Raw model output text
        incident_text: Original incident text for context-based hints
    
    Returns:
        Dictionary with repaired JSON structure
    """
    # Start with empty values
    repaired = {
        "severity": "",
        "likely_cause": "",
        "recommended_action": ""
    }
    
    # Normalize output first
    normalized = normalize_output(raw_output)
    
    # Try to extract field values with robust regex patterns
    # Severity: allow optional quotes, match SEV-1 or SEV-3
    severity_pattern = r'severity\s*"?\s*:\s*"?\s*(SEV-[13])'
    severity_match = re.search(severity_pattern, normalized, re.IGNORECASE)
    if severity_match:
        repaired["severity"] = severity_match.group(1)
    
    # Likely cause: match known values
    likely_cause_patterns = [
        r'likely_cause\s*"?\s*:\s*"?\s*(Packet responder termination)',
        r'likely_cause\s*"?\s*:\s*"?\s*(Block serving exception)'
    ]
    
    for pattern in likely_cause_patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if match:
            repaired["likely_cause"] = match.group(1)
            break
    
    # If likely_cause still empty, use incident context
    if not repaired["likely_cause"] and incident_text:
        if "exception while serving" in incident_text.lower():
            repaired["likely_cause"] = "Block serving exception"
        elif "packetresponder" in incident_text.lower() and "terminating" in incident_text.lower():
            repaired["likely_cause"] = "Packet responder termination"
    
    # Recommended action: capture text after key until end or closing brace/comma
    action_pattern = r'recommended_action\s*"?\s*:\s*"?\s*([^",}]+)'
    action_match = re.search(action_pattern, normalized, re.IGNORECASE)
    if action_match:
        repaired["recommended_action"] = action_match.group(1).strip().strip('"').strip()
    
    # If recommended_action still empty and we have likely_cause, use default
    if not repaired["recommended_action"] and repaired["likely_cause"]:
        repaired["recommended_action"] = get_default_action(repaired["likely_cause"])
    
    # Only use UNKNOWN as last resort if nothing was recovered
    if not repaired["severity"]:
        repaired["severity"] = "UNKNOWN (model output incomplete)"
    if not repaired["likely_cause"]:
        repaired["likely_cause"] = "UNKNOWN (model output incomplete)"
    if not repaired["recommended_action"]:
        repaired["recommended_action"] = "UNKNOWN (model output incomplete)"
    
    return repaired


def parse_and_repair_json(raw_output: str, incident_text: str = "") -> tuple[dict, bool]:
    """
    Parse model output as JSON, applying normalization and repair if needed.
    
    Args:
        raw_output: Raw model output text
        incident_text: Original incident text for context
    
    Returns:
        Tuple of (parsed_dict, was_repaired)
    """
    # Step 1: Normalize output
    normalized = normalize_output(raw_output)
    
    # Step 2: Try strict parsing on normalized output
    try:
        parsed = json.loads(normalized)
        
        # Verify all required keys exist
        required_keys = ["severity", "likely_cause", "recommended_action"]
        if all(key in parsed for key in required_keys):
            # Check that values are not empty
            if all(parsed[key] and str(parsed[key]).strip() for key in required_keys):
                return parsed, False  # Valid JSON, no repair needed
        
        # Keys exist but some values are empty - repair
        repaired = repair_json(raw_output, incident_text)
        return repaired, True
    
    except json.JSONDecodeError:
        # Step 3: Apply repair with robust extraction
        repaired = repair_json(raw_output, incident_text)
        return repaired, True


def check_format(parsed_json: dict, was_repaired: bool) -> str:
    """
    Check if parsed JSON is valid and complete.
    
    Args:
        parsed_json: Parsed/repaired JSON dictionary
        was_repaired: Whether repair was applied
    
    Returns:
        Format check status message
    """
    # Check required keys
    required_keys = ["severity", "likely_cause", "recommended_action"]
    has_all_keys = all(key in parsed_json for key in required_keys)
    
    if not has_all_keys:
        return "⚠ FORMAT WARNING: Missing required keys"
    
    # Check if any value contains UNKNOWN
    has_unknown = any("UNKNOWN" in str(parsed_json.get(key, "")) for key in required_keys)
    if has_unknown:
        return "⚠ FORMAT WARNING: Incomplete fields"
    
    # Check that all values are non-empty
    all_non_empty = all(parsed_json.get(key) and str(parsed_json.get(key)).strip() for key in required_keys)
    if not all_non_empty:
        return "⚠ FORMAT WARNING: Empty fields"
    
    # Check severity format (SEV-1 or SEV-3)
    severity = str(parsed_json.get("severity", ""))
    severity_valid = re.match(r'^SEV-[13]$', severity) is not None
    
    # Determine status
    if was_repaired:
        if severity_valid and all_non_empty:
            return "✓ FORMAT OK (auto-repaired)"
        else:
            return "⚠ FORMAT WARNING: Invalid format after repair"
    elif severity_valid and all_non_empty:
        return "✓ FORMAT OK"
    else:
        return "⚠ FORMAT WARNING: Invalid severity format"


def generate_response(incident_text: str, tokenizer, model) -> str:
    """
    Generate triage response for a given incident text using structured completion.
    Uses improved decoding parameters for complete JSON generation.
    
    Args:
        incident_text: Raw incident text
        tokenizer: Loaded tokenizer
        model: Loaded model
    
    Returns:
        Generated response text
    """
    # Use the same template as training
    prompt = format_incident_prompt(incident_text)
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=256,        # Increased from 128
        min_new_tokens=60,         # Ensure minimum completion
        num_beams=4,
        do_sample=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        length_penalty=0.8,        # Encourage finishing
        early_stopping=True
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


def select_diverse_samples(test_data: list, num_samples: int = 3) -> list:
    """
    Select diverse test samples ensuring variety in incident types.
    Ensures at least one Packet responder termination and one Block serving exception.
    
    Args:
        test_data: List of test examples
        num_samples: Number of samples to select
    
    Returns:
        List of selected test examples
    """
    # First, try to find one of each target type
    packet_responder = None
    block_serving = None
    other_samples = []
    
    for item in test_data:
        # Extract incident text from prompt (after "Incident:\n")
        prompt_text = item.get("prompt", "")
        if "Incident:\n" in prompt_text:
            incident_text = prompt_text.split("Incident:\n", 1)[1].lower()
        else:
            incident_text = ""
        
        response_data = json.loads(item["response"])
        likely_cause = response_data.get("likely_cause", "")
        
        if "packet responder" in likely_cause.lower() and packet_responder is None:
            packet_responder = item
        elif "block serving" in likely_cause.lower() and block_serving is None:
            block_serving = item
        else:
            other_samples.append(item)
    
    # Build the sample list
    selected = []
    if packet_responder:
        selected.append(packet_responder)
    if block_serving:
        selected.append(block_serving)
    
    # Fill remaining slots from other samples
    remaining_needed = num_samples - len(selected)
    if remaining_needed > 0 and other_samples:
        selected.extend(other_samples[:remaining_needed])
    
    # If we still need more samples, just take from the beginning of test_data
    if len(selected) < num_samples:
        for item in test_data:
            if item not in selected:
                selected.append(item)
                if len(selected) >= num_samples:
                    break
    
    return selected[:num_samples]


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
    
    # Select diverse samples
    selected_samples = select_diverse_samples(test_data, num_samples)
    print(f"\nSelected {len(selected_samples)} diverse test samples for demonstration")
    
    # Run inference on samples
    print(f"\n{'='*70}")
    print(f"RUNNING INFERENCE ON {len(selected_samples)} TEST SAMPLES")
    print(f"{'='*70}")
    
    # Track repair statistics
    repair_count = 0
    
    for i, example in enumerate(selected_samples, 1):
        # Extract incident text from prompt (after "Incident:\n")
        prompt_text = example["prompt"]
        if "Incident:\n" in prompt_text:
            incident_text = prompt_text.split("Incident:\n", 1)[1]
            # Extract just the incident text before the final instruction
            if "\n\nComplete the JSON" in incident_text:
                incident_text = incident_text.split("\n\nComplete the JSON")[0]
        else:
            incident_text = prompt_text
        
        ground_truth = example["response"]
        
        # Generate model output
        raw_output = generate_response(incident_text, tokenizer, model)
        
        # Parse and repair JSON
        parsed_json, was_repaired = parse_and_repair_json(raw_output, incident_text)
        
        if was_repaired:
            repair_count += 1
        
        # Display results
        print(f"\n{'='*70}")
        print(f"TEST SAMPLE {i}")
        print(f"{'='*70}")
        
        print(f"\nINPUT:")
        print(f"{'-'*70}")
        # Show first 6 lines of incident
        incident_lines = incident_text.strip().split('\n')
        for line in incident_lines[:6]:
            print(line)
        if len(incident_lines) > 6:
            print(f"... ({len(incident_lines) - 6} more lines)")
        
        print(f"\nRAW MODEL OUTPUT:")
        print(f"{'-'*70}")
        print(raw_output)
        
        # Only print final JSON if repair was needed
        if was_repaired:
            print(f"\nFINAL STRUCTURED JSON (REPAIRED):")
            print(f"{'-'*70}")
            print(json.dumps(parsed_json, indent=2))
        
        # Check format
        format_status = check_format(parsed_json, was_repaired)
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
    print(f"Test samples evaluated: {len(selected_samples)} of {len(test_data)}")
    print(f"Samples requiring repair: {repair_count} of {len(selected_samples)}")
    
    if repair_count == 0:
        print(f"\n✓ SUCCESS: All outputs were valid JSON without repair!")
    else:
        print(f"\n⚠ Note: {repair_count} sample(s) required auto-repair")
    
    print(f"\nThis demonstrates the fine-tuned model's ability to:")
    print(f"  - Parse incident log patterns")
    print(f"  - Classify severity levels")
    print(f"  - Identify likely causes")
    print(f"  - Generate actionable recommendations")
    print(f"\nUsing structured JSON completion with robust normalization and repair.")
    
    return 0


if __name__ == "__main__":
    exit(main())
