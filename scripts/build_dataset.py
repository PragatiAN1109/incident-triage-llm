#!/usr/bin/env python3
"""
Build fine-tuning dataset from preprocessed incidents.
Applies rule-based labeling for severity, likely_cause, and recommended_action.
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple


def assign_severity(incident: Dict) -> str:
    """
    Assign severity based on WARN/ERROR counts and keyword presence.
    
    Rules:
    - SEV-1 (Critical): any ERROR/FATAL OR critical keywords OR "exception while serving" OR WARN >= 4
    - SEV-2 (High): WARN count 2-3
    - SEV-3 (Low): WARN count 0-1
    """
    levels_count = incident["stats"]["levels_count"]
    warn_count = levels_count.get("WARN", 0)
    error_count = levels_count.get("ERROR", 0)
    fatal_count = levels_count.get("FATAL", 0)
    incident_text_lower = incident["incident_text"].lower()
    
    # Check for critical keywords
    critical_keywords = ["exception", "fatal", "error", "failed"]
    has_critical = any(kw in incident_text_lower for kw in critical_keywords)
    
    # SEV-1: any ERROR/FATAL OR critical keywords OR "exception while serving" OR WARN >= 4
    if error_count > 0 or fatal_count > 0 or has_critical or "exception while serving" in incident_text_lower or warn_count >= 4:
        return "SEV-1"
    elif warn_count >= 2:
        return "SEV-2"
    else:
        return "SEV-3"


def assign_likely_cause(incident: Dict) -> str:
    """
    Infer likely cause from incident text keywords.
    Uses strict priority order (first match wins).
    
    Priority order:
    1. Block serving exception
    2. Network connectivity issue
    3. Service degradation
    4. Packet responder termination
    5. HDFS operational event (default)
    """
    incident_text_lower = incident["incident_text"].lower()
    
    # Priority 1: Block serving exception
    if "exception while serving" in incident_text_lower:
        return "Block serving exception"
    
    # Priority 2: Network connectivity issue
    network_keywords = ["timeout", "refused", "unreachable", "disconnect"]
    if any(kw in incident_text_lower for kw in network_keywords):
        return "Network connectivity issue"
    
    # Priority 3: Service degradation
    degradation_keywords = ["slow", "lag", "backlog", "thrott"]
    if any(kw in incident_text_lower for kw in degradation_keywords):
        return "Service degradation"
    
    # Priority 4: Packet responder termination
    if "packetresponder" in incident_text_lower and "terminating" in incident_text_lower:
        return "Packet responder termination"
    
    # Priority 5: Default
    return "HDFS operational event"


def assign_recommended_action(likely_cause: str) -> str:
    """
    Map likely cause to recommended action.
    """
    action_map = {
        "Block serving exception": "Investigate DataNode block serving failures and check network connectivity between nodes",
        "Packet responder termination": "Monitor DataNode for abnormal packet responder patterns; verify normal block replication",
        "Network connectivity issue": "Check network configuration and firewall rules; verify DataNode-NameNode connectivity",
        "Service degradation": "Review system resources (CPU, memory, disk I/O); check for performance bottlenecks",
        "HDFS operational event": "Monitor cluster health metrics; no immediate action required unless pattern persists"
    }
    
    return action_map.get(likely_cause, "Review incident logs and correlate with cluster metrics")


def label_incident(incident: Dict) -> Dict:
    """
    Add labels to an incident.
    Returns prompt and response as a JSON string (NOT double-encoded).
    """
    severity = assign_severity(incident)
    likely_cause = assign_likely_cause(incident)
    recommended_action = assign_recommended_action(likely_cause)
    
    response_obj = {
        "severity": severity,
        "likely_cause": likely_cause,
        "recommended_action": recommended_action
    }
    
    # Convert response object to JSON string (single level of encoding)
    response_str = json.dumps(response_obj)
    
    return {
        "prompt": incident["incident_text"],
        "response": response_str  # This is a plain string containing JSON
    }


def validate_response_format(labeled_incidents: List[Dict], num_samples: int = 3):
    """
    Validate that response strings can be parsed as JSON.
    Samples random examples to verify correct encoding.
    """
    print(f"\nValidating response format (sampling {num_samples} examples)...")
    
    if len(labeled_incidents) < num_samples:
        num_samples = len(labeled_incidents)
    
    sample_indices = random.sample(range(len(labeled_incidents)), num_samples)
    
    for idx in sample_indices:
        item = labeled_incidents[idx]
        try:
            # Try to parse the response string as JSON
            parsed = json.loads(item["response"])
            
            # Verify required fields
            required_fields = ["severity", "likely_cause", "recommended_action"]
            missing = [f for f in required_fields if f not in parsed]
            
            if missing:
                print(f"  ✗ Sample {idx}: Missing fields: {missing}")
                print(f"    Response: {item['response']}")
                return False
            
            print(f"  ✓ Sample {idx}: Valid JSON with all required fields")
            
        except json.JSONDecodeError as e:
            print(f"  ✗ Sample {idx}: JSON parse error: {e}")
            print(f"    Response: {item['response']}")
            return False
    
    print(f"✓ All samples validated successfully")
    return True


def split_dataset(
    labeled_incidents: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset into train/val/test with stratification by severity.
    """
    random.seed(seed)
    
    # Group by severity for stratified split
    severity_groups = {}
    for item in labeled_incidents:
        response_data = json.loads(item["response"])
        severity = response_data["severity"]
        if severity not in severity_groups:
            severity_groups[severity] = []
        severity_groups[severity].append(item)
    
    train_data = []
    val_data = []
    test_data = []
    
    # Split each severity group proportionally
    for severity, items in severity_groups.items():
        random.shuffle(items)
        n = len(items)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_data.extend(items[:train_end])
        val_data.extend(items[train_end:val_end])
        test_data.extend(items[val_end:])
    
    # Shuffle the final splits
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data


def print_split_stats(train_data: List[Dict], val_data: List[Dict], test_data: List[Dict]):
    """
    Print detailed statistics for each split.
    """
    def get_severity_dist(data):
        counts = Counter()
        for item in data:
            response_data = json.loads(item["response"])
            counts[response_data["severity"]] += 1
        return counts
    
    print(f"\n{'='*60}")
    print(f"SPLIT STATISTICS")
    print(f"{'='*60}")
    
    total = len(train_data) + len(val_data) + len(test_data)
    print(f"Total samples: {total}")
    
    print(f"\nTrain: {len(train_data)} samples")
    train_sev = get_severity_dist(train_data)
    for sev in sorted(train_sev.keys()):
        print(f"  {sev}: {train_sev[sev]}")
    
    print(f"\nValidation: {len(val_data)} samples")
    val_sev = get_severity_dist(val_data)
    for sev in sorted(val_sev.keys()):
        print(f"  {sev}: {val_sev[sev]}")
    
    print(f"\nTest: {len(test_data)} samples")
    test_sev = get_severity_dist(test_data)
    for sev in sorted(test_sev.keys()):
        print(f"  {sev}: {test_sev[sev]}")
    
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Build fine-tuning dataset from preprocessed incidents"
    )
    parser.add_argument(
        "--input",
        default="data/processed/sample_incidents_50.jsonl",
        help="Input preprocessed incidents file"
    )
    parser.add_argument(
        "--output_dir",
        default="data/final",
        help="Output directory for train/val/test splits"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test set ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Error: Ratios must sum to 1.0 (got {total_ratio})")
        return 1
    
    # Read input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    print(f"Reading incidents from: {args.input}")
    incidents = []
    with open(input_path, 'r') as f:
        for line in f:
            incidents.append(json.loads(line))
    
    print(f"Total incidents: {len(incidents)}")
    
    # Apply labeling
    print("\nApplying rule-based labels...")
    labeled_incidents = [label_incident(inc) for inc in incidents]
    
    # Validate response format
    if not validate_response_format(labeled_incidents):
        print("\nError: Response format validation failed!")
        return 1
    
    # Show label distribution
    severity_counts = Counter()
    cause_counts = Counter()
    
    for item in labeled_incidents:
        response_data = json.loads(item["response"])
        severity_counts[response_data["severity"]] += 1
        cause_counts[response_data["likely_cause"]] += 1
    
    print(f"\nLabel distribution:")
    print(f"  Severity:")
    for sev, count in sorted(severity_counts.items()):
        print(f"    {sev}: {count} ({count/len(labeled_incidents)*100:.1f}%)")
    
    print(f"  Likely Cause:")
    for cause, count in cause_counts.most_common():
        print(f"    {cause}: {count}")
    
    # Split dataset
    print(f"\nSplitting dataset (train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio})...")
    train_data, val_data, test_data = split_dataset(
        labeled_incidents,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )
    
    # Print detailed split statistics
    print_split_stats(train_data, val_data, test_data)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write splits
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"
    
    print(f"\nWriting splits...")
    with open(train_path, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    print(f"  Train: {train_path}")
    
    with open(val_path, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    print(f"  Validation: {val_path}")
    
    with open(test_path, 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    print(f"  Test: {test_path}")
    
    print(f"\n✓ Dataset build complete!")
    print(f"  Output directory: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
