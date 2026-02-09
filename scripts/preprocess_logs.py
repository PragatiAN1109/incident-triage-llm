#!/usr/bin/env python3
"""
Log preprocessing script for incident triage.
Cleans, filters, normalizes, and groups log lines into incidents.
"""

import argparse
import json
import re
import hashlib
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional


def parse_log_line(line: str) -> Dict[str, str]:
    """
    Extract log level, component, and message from a log line.
    Best effort parsing - handles variations in log format.
    """
    line = line.strip()
    if not line:
        return {"level": "INFO", "component": "unknown", "message": line}
    
    # Try to find log level (INFO, WARN, ERROR, FATAL, DEBUG)
    level_match = re.search(r'\b(INFO|WARN|ERROR|FATAL|DEBUG)\b', line)
    level = level_match.group(1) if level_match else "INFO"
    
    # Try to extract component (looks for patterns like dfs.DataNode, FSNamesystem, etc.)
    component_match = re.search(r'\b(dfs\.\w+|\w+\.\w+)\b', line)
    component = component_match.group(1) if component_match else "unknown"
    
    # Extract message (everything after the log level, or the whole line)
    if level_match:
        message = line[level_match.end():].strip()
        # Remove leading colons
        message = message.lstrip(':').strip()
    else:
        message = line
    
    return {
        "level": level,
        "component": component,
        "message": message
    }


def normalize_text(text: str) -> str:
    """
    Normalize text by:
    - Converting to lowercase
    - Stripping leading timestamps and numeric IDs
    - Collapsing whitespace
    - Replacing block IDs (blk_####) with blk_<ID>
    """
    # Lowercase
    text = text.lower()
    
    # Remove leading timestamps (e.g., 081109 203615 148)
    text = re.sub(r'^\d{6}\s+\d{6}\s+\d+\s*', '', text)
    
    # Replace block IDs with placeholder
    text = re.sub(r'blk_-?\d+', 'blk_<ID>', text)
    
    # Replace IP addresses with placeholder (optional but helpful)
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+\b', '<IP:PORT>', text)
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', text)
    
    # Replace size numbers
    text = re.sub(r'\bsize\s+\d+\b', 'size <SIZE>', text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def is_informative(parsed: Dict[str, str]) -> bool:
    """
    Determine if a log line is informative enough to keep.
    Keep WARN/ERROR/FATAL, or INFO/DEBUG with specific keywords.
    """
    level = parsed["level"]
    message = parsed["message"].lower()
    
    # Always keep warnings, errors, and fatal
    if level in ["WARN", "ERROR", "FATAL"]:
        return True
    
    # Keep INFO/DEBUG with specific keywords
    keywords = [
        "retry", "slow", "timeout", "fail", "exception", "terminating",
        "unable", "refused", "exceeded", "corrupt", "denied", "backlog",
        "lag", "waiting", "dropped", "disconnect", "unreachable",
        "exhausted", "thrott"
    ]
    
    return any(keyword in message for keyword in keywords)


def infer_service(component: str) -> str:
    """
    Infer service name from component.
    """
    component_lower = component.lower()
    
    if "datanode" in component_lower:
        return "hdfs-datanode"
    elif "namenode" in component_lower or "namesystem" in component_lower:
        return "hdfs-namenode"
    else:
        return "hdfs"


def group_into_incidents(
    lines: List[Tuple[str, Dict[str, str]]],
    group_size: int,
    seed: int,
    max_incidents: Optional[int] = None
) -> List[Dict]:
    """
    Group log lines into incidents using a rolling buffer.
    Each incident contains group_size consecutive informative lines.
    Buffer is cleared after each incident to prevent duplicates.
    """
    incidents = []
    buffer = []
    incident_index = 0
    
    for raw_line, parsed in lines:
        buffer.append((raw_line, parsed))
        
        if len(buffer) >= group_size:
            # Create incident from buffer
            incident_lines = [line for line, _ in buffer[:group_size]]
            parsed_lines = [p for _, p in buffer[:group_size]]
            
            # Infer service from most common component
            components = [p["component"] for p in parsed_lines]
            most_common_component = Counter(components).most_common(1)[0][0]
            service = infer_service(most_common_component)
            
            # Create incident_text with service prefix
            lines_text = "\n".join(incident_lines)
            incident_text = f"service: {service}\n{lines_text}"
            
            # Create incident ID using sha1 hash of incident_text + index for uniqueness
            hash_input = f"{incident_text}|{incident_index}"
            hash_obj = hashlib.sha1(hash_input.encode('utf-8'))
            incident_id = hash_obj.hexdigest()[:12]
            
            # Calculate stats
            levels_count = dict(Counter(p["level"] for p in parsed_lines))
            top_components = dict(Counter(components).most_common(3))
            
            incident = {
                "incident_id": incident_id,
                "service": service,
                "lines": incident_lines,
                "incident_text": incident_text,
                "stats": {
                    "levels_count": levels_count,
                    "top_components": top_components
                }
            }
            
            incidents.append(incident)
            incident_index += 1
            
            # Clear the buffer completely to avoid duplicates
            buffer = []
            
            # Check if we've reached max incidents
            if max_incidents and len(incidents) >= max_incidents:
                break
    
    return incidents


def process_logs(
    input_path: str,
    output_path: str,
    group_size: int = 8,
    seed: int = 42,
    max_incidents: Optional[int] = None,
    dry_run: bool = False
) -> Dict:
    """
    Main processing function.
    """
    input_file = Path(input_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read and parse log lines
    print(f"Reading log file: {input_path}")
    informative_lines = []
    total_lines = 0
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            
            parsed = parse_log_line(line)
            
            if is_informative(parsed):
                normalized_line = normalize_text(line)
                informative_lines.append((normalized_line, parsed))
    
    print(f"Total lines read: {total_lines}")
    print(f"Informative lines: {len(informative_lines)}")
    
    # Dry run - show samples and summary
    if dry_run:
        print("\n=== DRY RUN MODE ===")
        print(f"\nSample of 5 normalized informative records:")
        for i, (norm_line, parsed) in enumerate(informative_lines[:5], 1):
            print(f"\n[{i}] Level: {parsed['level']}, Component: {parsed['component']}")
            print(f"    Normalized: {norm_line}")
        
        print(f"\n=== Summary ===")
        print(f"Would generate approximately {len(informative_lines) // group_size} incidents")
        print(f"Group size: {group_size}")
        if max_incidents:
            print(f"Max incidents limit: {max_incidents}")
        
        return {
            "total_lines": total_lines,
            "informative_lines": len(informative_lines),
            "incidents_generated": 0,
            "dry_run": True
        }
    
    # Group into incidents
    print(f"\nGrouping into incidents (group_size={group_size})...")
    incidents = group_into_incidents(informative_lines, group_size, seed, max_incidents)
    
    # Verify uniqueness
    incident_ids = [inc["incident_id"] for inc in incidents]
    unique_ids = set(incident_ids)
    if len(unique_ids) != len(incident_ids):
        print(f"WARNING: Found {len(incident_ids) - len(unique_ids)} duplicate incident IDs!")
    else:
        print(f"✓ All {len(incidents)} incident IDs are unique")
    
    # Write output
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Writing {len(incidents)} incidents to: {output_path}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for incident in incidents:
            f.write(json.dumps(incident) + '\n')
    
    print(f"\n=== Summary ===")
    print(f"Total lines read: {total_lines}")
    print(f"Informative lines: {len(informative_lines)}")
    print(f"Incidents generated: {len(incidents)}")
    print(f"Unique incident IDs: {len(unique_ids)}")
    print(f"Output written to: {output_path}")
    
    return {
        "total_lines": total_lines,
        "informative_lines": len(informative_lines),
        "incidents_generated": len(incidents),
        "output_path": output_path
    }


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess HDFS logs into incident groups"
    )
    parser.add_argument(
        "--input",
        default="data/raw/HDFS_2k.log",
        help="Input log file path"
    )
    parser.add_argument(
        "--output",
        default="data/processed/incidents.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=8,
        help="Number of log lines per incident group"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--max_incidents",
        type=int,
        default=500,
        help="Maximum number of incidents to generate"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show summary without writing output"
    )
    
    args = parser.parse_args()
    
    try:
        process_logs(
            input_path=args.input,
            output_path=args.output,
            group_size=args.group_size,
            seed=args.seed,
            max_incidents=args.max_incidents,
            dry_run=args.dry_run
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
