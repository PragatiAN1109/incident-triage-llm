#!/usr/bin/env python3
"""
Shared prompt template for incident triage task.
Ensures consistent formatting between training and inference.
"""


def format_incident_prompt(incident_text: str) -> str:
    """
    Format incident text using structured JSON completion template.
    
    This template is used for BOTH training and inference to ensure consistency.
    The slot-filling approach helps the model learn to complete specific fields
    rather than generate schema from scratch, which is more stable on small datasets.
    
    Args:
        incident_text: Raw incident log text
    
    Returns:
        Formatted prompt with JSON template
    """
    prompt = f"""You are an incident triage assistant.
Fill in the JSON template below using the incident information.

JSON:
{{
  "severity": "",
  "likely_cause": "",
  "recommended_action": ""
}}

Incident:
{incident_text}

Output MUST be valid JSON object with exactly these 3 keys.
Do not output partial JSON. Do not stop early.
severity must be one of: SEV-1 or SEV-3.
likely_cause must be one of: Packet responder termination OR Block serving exception.

Output ONLY the completed JSON:
"""
    return prompt
