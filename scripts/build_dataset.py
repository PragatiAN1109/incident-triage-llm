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
    - SEV-1 (Critical): 6+ WARN or any ERROR, or keywords like "exception"
    - SEV-2 (High): 3-5 WARN
    - SEV-3 (Low): 0-2 WARN (mostly INFO)
    """
    levels_count = incident["stats"]["levels_count"]
    warn_count = levels_count.get("WARN", 0)
    error_count = levels_count.get("ERROR", 0)
    incident_text_lower = incident["incident_text"].lower()
    
    # Check for critical keywords
    critical_keywords = ["exception", "error", "fatal", "failed", "refused", "denied"]
    has_critical = any(kw in incident_text_lower for kw in critical_keywords)
    
    if error_count > 0 or warn_count >= 6 or has_critical:
        return "SEV-1"
    elif warn_count >= 3:
        return "SEV-2"
    else:
        return "SEV-3"


def assign_likely_cause(incident: Dict) -> str:
    """
    Infer likely cause from incident text keywords.
    
    Categories:
    - Block serving exception
    - Packet responder termination
    - Network connectivity issue
    - Service degradation
    """
    incident_text_lower = incident["incident_text"].lower()
    
    # Check for specific patterns
    if "exception while serving" in incident_text_lower:
        return "Block serving exception"
    elif "packetresponder" in incident_text_lower and "terminating" in incident_text_lower:
        return "Packet responder termination"
    elif "timeout" in incident_text_lower or "refused" in incident_text_lower:
        return "Network connectivity issue"
    elif "slow" in incident_text_lower or "lag" in incident_text_lower:
        return "Service degradation"
    else:
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
    """
    severity = assign_severity(incident)
    likely_cause = assign_likely_cause(incident)
    recommended_action = assign_recommended_action(likely_cause)
    
    response = {
        "severity": severity,
        "likely_cause": likely_cause,
        "recommended_action": recommended_action
    }
    
    return {
        "prompt": incident["incident_text"],
        "response": json.dumps(response)
    }


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
    
    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
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
