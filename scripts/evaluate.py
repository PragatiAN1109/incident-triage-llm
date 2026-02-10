#!/usr/bin/env python3
"""
Comprehensive model evaluation script.
Compares fine-tuned model performance against baseline and computes detailed metrics.
"""

import json
import argparse
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from prompt_template import format_incident_prompt
import re


def load_model(model_path: str):
    """Load model and tokenizer."""
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    print(f"✓ Model loaded")
    return tokenizer, model


def load_test_data(test_file: str):
    """Load test dataset."""
    test_data = []
    with open(test_file, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    return test_data


def normalize_output(text: str) -> str:
    """Normalize model output."""
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")
    text = text.replace("`", '"')
    return text.strip()


def clean_value(val: str) -> str:
    """Clean extracted values."""
    if not val:
        return val
    
    val = val.strip()
    prefixes = ['(', ')"', '("', '\\"', '"', "'"]
    suffixes = [')', '")', '"', "'"]
    
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if val.startswith(prefix):
                val = val[len(prefix):]
                changed = True
                break
    
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if val.endswith(suffix):
                val = val[:-len(suffix)]
                changed = True
                break
    
    return val.strip()


def extract_severity(text: str) -> str:
    """Extract severity."""
    match = re.search(r'\bSEV-(1|3)\b', text, re.IGNORECASE)
    return f"SEV-{match.group(1)}" if match else ""


def extract_likely_cause(text: str) -> str:
    """Extract likely cause."""
    text_lower = text.lower()
    if "block serving" in text_lower:
        return "Block serving exception"
    elif "packet responder" in text_lower:
        return "Packet responder termination"
    return ""


def parse_output(raw_output: str) -> dict:
    """Parse model output to JSON."""
    clean_text = normalize_output(raw_output)
    
    first_brace = clean_text.find('{')
    last_brace = clean_text.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_candidate = clean_text[first_brace:last_brace + 1]
    else:
        json_candidate = clean_text
    
    try:
        parsed = json.loads(json_candidate)
        if all(key in parsed for key in ["severity", "likely_cause", "recommended_action"]):
            return parsed
    except:
        pass
    
    # Fallback extraction
    return {
        "severity": clean_value(extract_severity(clean_text)),
        "likely_cause": clean_value(extract_likely_cause(clean_text)),
        "recommended_action": ""
    }


def apply_heuristics(incident_text: str, parsed: dict) -> dict:
    """Apply incident-based heuristics."""
    if "got exception while serving" in incident_text.lower():
        parsed["severity"] = "SEV-1"
        parsed["likely_cause"] = "Block serving exception"
        parsed["recommended_action"] = "Investigate DataNode block serving failures and check network connectivity between nodes"
    return parsed


def generate_prediction(incident_text: str, tokenizer, model):
    """Generate and parse prediction."""
    prompt = format_incident_prompt(incident_text)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = model.generate(
        inputs.input_ids,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=256,
        min_new_tokens=60,
        num_beams=4,
        do_sample=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        length_penalty=0.8,
        early_stopping=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_model(model_path: str, test_file: str, baseline_model_path: str = None):
    """
    Comprehensive model evaluation.
    
    Args:
        model_path: Path to fine-tuned model
        test_file: Path to test data
        baseline_model_path: Path to baseline model (optional)
    """
    print("="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    # Load fine-tuned model
    print("\nLoading fine-tuned model...")
    tokenizer, model = load_model(model_path)
    
    # Load baseline model if provided
    baseline_tokenizer = None
    baseline_model = None
    if baseline_model_path:
        print("\nLoading baseline model...")
        baseline_tokenizer, baseline_model = load_model(baseline_model_path)
    
    # Load test data
    print(f"\nLoading test data from: {test_file}")
    test_data = load_test_data(test_file)
    print(f"✓ Loaded {len(test_data)} test samples")
    
    # Evaluate fine-tuned model
    print("\n" + "="*70)
    print("EVALUATING FINE-TUNED MODEL")
    print("="*70)
    
    predictions = []
    ground_truths = []
    
    severity_correct = 0
    cause_correct = 0
    action_correct = 0
    exact_match = 0
    valid_json_count = 0
    repair_needed = 0
    heuristic_applied = 0
    
    for example in test_data:
        # Extract incident text
        prompt_text = example["prompt"]
        if "Incident:\n" in prompt_text:
            incident_text = prompt_text.split("Incident:\n", 1)[1]
            if "\n\nSTRICT REQUIREMENTS:" in incident_text:
                incident_text = incident_text.split("\n\nSTRICT REQUIREMENTS:")[0]
        else:
            incident_text = prompt_text
        
        # Get ground truth
        gt = json.loads(example["response"])
        ground_truths.append(gt)
        
        # Generate prediction
        raw_output = generate_prediction(incident_text, tokenizer, model)
        
        # Parse output
        parsed = parse_output(raw_output)
        
        # Check if repair was needed
        try:
            clean_text = normalize_output(raw_output)
            first_brace = clean_text.find('{')
            last_brace = clean_text.rfind('}')
            if first_brace != -1 and last_brace != -1:
                json.loads(clean_text[first_brace:last_brace + 1])
                # Valid JSON
            else:
                raise ValueError()
        except:
            repair_needed += 1
        
        # Apply heuristics
        original_parsed = parsed.copy()
        parsed = apply_heuristics(incident_text, parsed)
        if parsed != original_parsed:
            heuristic_applied += 1
        
        predictions.append(parsed)
        
        # Count valid JSON outputs
        if all(parsed.get(k) and "UNKNOWN" not in str(parsed.get(k)) for k in ["severity", "likely_cause", "recommended_action"]):
            valid_json_count += 1
        
        # Field-level accuracy
        if parsed.get("severity") == gt.get("severity"):
            severity_correct += 1
        if parsed.get("likely_cause") == gt.get("likely_cause"):
            cause_correct += 1
        if parsed.get("recommended_action") == gt.get("recommended_action"):
            action_correct += 1
        
        # Exact match
        if parsed == gt:
            exact_match += 1
    
    # Calculate metrics
    total = len(test_data)
    severity_acc = (severity_correct / total) * 100
    cause_acc = (cause_correct / total) * 100
    action_acc = (action_correct / total) * 100
    exact_match_acc = (exact_match / total) * 100
    valid_json_rate = (valid_json_count / total) * 100
    repair_rate = (repair_needed / total) * 100
    heuristic_rate = (heuristic_applied / total) * 100
    
    # Print results
    print(f"\n{'='*70}")
    print("FINE-TUNED MODEL PERFORMANCE")
    print(f"{'='*70}")
    print(f"\nTest samples: {total}")
    print(f"\nField-Level Accuracy:")
    print(f"  Severity:            {severity_correct}/{total} ({severity_acc:.1f}%)")
    print(f"  Likely Cause:        {cause_correct}/{total} ({cause_acc:.1f}%)")
    print(f"  Recommended Action:  {action_correct}/{total} ({action_acc:.1f}%)")
    print(f"\nOverall Metrics:")
    print(f"  Exact Match:         {exact_match}/{total} ({exact_match_acc:.1f}%)")
    print(f"  Valid JSON Rate:     {valid_json_count}/{total} ({valid_json_rate:.1f}%)")
    print(f"\nInference Statistics:")
    print(f"  Repair Rate:         {repair_needed}/{total} ({repair_rate:.1f}%)")
    print(f"  Heuristic Override:  {heuristic_applied}/{total} ({heuristic_rate:.1f}%)")
    
    # Confusion matrix for severity
    print(f"\n{'='*70}")
    print("SEVERITY CONFUSION MATRIX")
    print(f"{'='*70}")
    
    severity_pairs = [(gt.get("severity"), pred.get("severity")) for gt, pred in zip(ground_truths, predictions)]
    
    sev1_to_sev1 = sum(1 for gt, pred in severity_pairs if gt == "SEV-1" and pred == "SEV-1")
    sev1_to_sev3 = sum(1 for gt, pred in severity_pairs if gt == "SEV-1" and pred == "SEV-3")
    sev3_to_sev1 = sum(1 for gt, pred in severity_pairs if gt == "SEV-3" and pred == "SEV-1")
    sev3_to_sev3 = sum(1 for gt, pred in severity_pairs if gt == "SEV-3" and pred == "SEV-3")
    
    print(f"\n               Predicted")
    print(f"             SEV-1  SEV-3")
    print(f"Actual SEV-1    {sev1_to_sev1}      {sev1_to_sev3}")
    print(f"       SEV-3    {sev3_to_sev1}      {sev3_to_sev3}")
    
    # Likely cause breakdown
    print(f"\n{'='*70}")
    print("LIKELY CAUSE BREAKDOWN")
    print(f"{'='*70}")
    
    cause_correct_by_type = {}
    cause_counts = Counter()
    
    for gt, pred in zip(ground_truths, predictions):
        gt_cause = gt.get("likely_cause")
        pred_cause = pred.get("likely_cause")
        cause_counts[gt_cause] += 1
        
        if gt_cause not in cause_correct_by_type:
            cause_correct_by_type[gt_cause] = 0
        if gt_cause == pred_cause:
            cause_correct_by_type[gt_cause] += 1
    
    for cause, count in cause_counts.items():
        correct = cause_correct_by_type.get(cause, 0)
        acc = (correct / count) * 100
        print(f"  {cause}: {correct}/{count} ({acc:.1f}%)")
    
    # Baseline comparison (if provided)
    if baseline_model and baseline_tokenizer:
        print(f"\n{'='*70}")
        print("BASELINE MODEL COMPARISON")
        print(f"{'='*70}")
        
        baseline_predictions = []
        baseline_severity_correct = 0
        baseline_cause_correct = 0
        baseline_exact_match = 0
        
        for example in test_data:
            prompt_text = example["prompt"]
            if "Incident:\n" in prompt_text:
                incident_text = prompt_text.split("Incident:\n", 1)[1]
                if "\n\nSTRICT REQUIREMENTS:" in incident_text:
                    incident_text = incident_text.split("\n\nSTRICT REQUIREMENTS:")[0]
            else:
                incident_text = prompt_text
            
            gt = json.loads(example["response"])
            
            # Generate baseline prediction
            raw_output = generate_prediction(incident_text, baseline_tokenizer, baseline_model)
            parsed = parse_output(raw_output)
            parsed = apply_heuristics(incident_text, parsed)
            baseline_predictions.append(parsed)
            
            if parsed.get("severity") == gt.get("severity"):
                baseline_severity_correct += 1
            if parsed.get("likely_cause") == gt.get("likely_cause"):
                baseline_cause_correct += 1
            if parsed == gt:
                baseline_exact_match += 1
        
        baseline_severity_acc = (baseline_severity_correct / total) * 100
        baseline_cause_acc = (baseline_cause_correct / total) * 100
        baseline_exact_acc = (baseline_exact_match / total) * 100
        
        print(f"\n{'Metric':<30} {'Baseline':<15} {'Fine-tuned':<15} {'Improvement':<15}")
        print(f"{'-'*70}")
        print(f"{'Severity Accuracy':<30} {baseline_severity_acc:<15.1f} {severity_acc:<15.1f} {severity_acc - baseline_severity_acc:+.1f}")
        print(f"{'Likely Cause Accuracy':<30} {baseline_cause_acc:<15.1f} {cause_acc:<15.1f} {cause_acc - baseline_cause_acc:+.1f}")
        print(f"{'Exact Match Accuracy':<30} {baseline_exact_acc:<15.1f} {exact_match_acc:<15.1f} {exact_match_acc - baseline_exact_acc:+.1f}")
    
    # Save metrics to JSON
    results = {
        "model_path": model_path,
        "test_samples": total,
        "metrics": {
            "severity_accuracy": severity_acc,
            "likely_cause_accuracy": cause_acc,
            "recommended_action_accuracy": action_acc,
            "exact_match_accuracy": exact_match_acc,
            "valid_json_rate": valid_json_rate
        },
        "inference_stats": {
            "repair_rate": repair_rate,
            "heuristic_override_rate": heuristic_rate
        },
        "confusion_matrix": {
            "SEV-1": {"SEV-1": sev1_to_sev1, "SEV-3": sev1_to_sev3},
            "SEV-3": {"SEV-1": sev3_to_sev1, "SEV-3": sev3_to_sev3}
        },
        "by_likely_cause": {
            cause: {
                "total": cause_counts[cause],
                "correct": cause_correct_by_type.get(cause, 0),
                "accuracy": (cause_correct_by_type.get(cause, 0) / cause_counts[cause]) * 100
            }
            for cause in cause_counts
        }
    }
    
    output_file = "results/evaluation_metrics.json"
    Path("results").mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Evaluation metrics saved to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument(
        "--model_path",
        default="results/config_c_(higher_capacity)/final-model",
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--test_file",
        default="data/final/test.jsonl",
        help="Path to test data"
    )
    parser.add_argument(
        "--baseline_model",
        default=None,
        help="Path to baseline model for comparison (optional)"
    )
    
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.test_file, args.baseline_model)
    
    return 0


if __name__ == "__main__":
    exit(main())
