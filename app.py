import gradio as gr
import json
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer
MODEL_PATH = "PragatiAN1109/incident-triage-llm"  # Will be your HF model repo
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
print("✓ Model loaded")


def format_incident_prompt(incident_text: str) -> str:
    """Format incident using structured JSON completion template."""
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

STRICT REQUIREMENTS:
- Return a SINGLE JSON object only. No extra text.
- Use ONLY standard ASCII double quotes ("). Do NOT use smart quotes.
- Do NOT use parentheses for values.
- Do NOT stop early. Ensure recommended_action is present and non-empty.
- severity must be exactly "SEV-1" or "SEV-3"
- likely_cause must be exactly "Packet responder termination" OR "Block serving exception"

Output ONLY the completed JSON:
"""
    return prompt


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


def get_default_action(likely_cause: str) -> str:
    """Get default action."""
    defaults = {
        "Packet responder termination": "Monitor DataNode for abnormal packet responder patterns; verify normal block replication",
        "Block serving exception": "Investigate DataNode block serving failures and check network connectivity between nodes"
    }
    return defaults.get(likely_cause, "Review incident logs and correlate with cluster metrics")


def parse_and_repair_json(raw_output: str) -> dict:
    """Parse and repair model output."""
    clean_text = normalize_output(raw_output)
    
    first_brace = clean_text.find('{')
    last_brace = clean_text.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_candidate = clean_text[first_brace:last_brace + 1]
    else:
        json_candidate = clean_text
    
    try:
        parsed = json.loads(json_candidate)
        required_keys = ["severity", "likely_cause", "recommended_action"]
        if all(key in parsed for key in required_keys):
            all_valid = all(
                parsed[key] and 
                str(parsed[key]).strip() and 
                "UNKNOWN" not in str(parsed[key])
                for key in required_keys
            )
            if all_valid:
                return parsed
    except:
        pass
    
    # Fallback extraction
    repaired = {
        "severity": clean_value(extract_severity(clean_text)),
        "likely_cause": clean_value(extract_likely_cause(clean_text)),
        "recommended_action": ""
    }
    
    if not repaired["recommended_action"] and repaired["likely_cause"]:
        repaired["recommended_action"] = get_default_action(repaired["likely_cause"])
    
    if not repaired["severity"]:
        repaired["severity"] = "SEV-3"
    if not repaired["likely_cause"]:
        repaired["likely_cause"] = "HDFS operational event"
    if not repaired["recommended_action"]:
        repaired["recommended_action"] = "Review incident logs and correlate with cluster metrics"
    
    return repaired


def apply_heuristics(incident_text: str, parsed: dict) -> dict:
    """Apply incident-based heuristics."""
    if "got exception while serving" in incident_text.lower():
        parsed["severity"] = "SEV-1"
        parsed["likely_cause"] = "Block serving exception"
        parsed["recommended_action"] = "Investigate DataNode block serving failures and check network connectivity between nodes"
    return parsed


def triage_incident(incident_logs: str):
    """
    Main triage function for Gradio interface.
    
    Args:
        incident_logs: Raw incident log text
    
    Returns:
        Tuple of (severity, likely_cause, recommended_action, raw_output, confidence)
    """
    if not incident_logs.strip():
        return "N/A", "N/A", "Please enter incident logs", "", "N/A"
    
    # Generate prediction
    prompt = format_incident_prompt(incident_logs)
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
    
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse and repair
    parsed = parse_and_repair_json(raw_output)
    parsed = apply_heuristics(incident_logs, parsed)
    
    # Determine confidence
    try:
        clean = normalize_output(raw_output)
        json.loads(clean[clean.find('{'):clean.rfind('}')+1])
        confidence = "High (Valid JSON)"
    except:
        confidence = "Medium (Auto-repaired)"
    
    return (
        parsed.get("severity", "N/A"),
        parsed.get("likely_cause", "N/A"),
        parsed.get("recommended_action", "N/A"),
        raw_output,
        confidence
    )


# Example incidents
EXAMPLE_1 = """service: hdfs-datanode
info dfs.datanode$packetresponder: packetresponder 0 for block blk_<ID> terminating
info dfs.datanode$packetresponder: packetresponder 0 for block blk_<ID> terminating
info dfs.datanode$packetresponder: packetresponder 2 for block blk_<ID> terminating
info dfs.datanode$packetresponder: packetresponder 1 for block blk_<ID> terminating
info dfs.datanode$packetresponder: packetresponder 0 for block blk_<ID> terminating"""

EXAMPLE_2 = """service: hdfs-datanode
warn dfs.datanode$dataxceiver: <IP:PORT>:got exception while serving blk_<ID> to /<IP>:
warn dfs.datanode$dataxceiver: <IP:PORT>:got exception while serving blk_<ID> to /<IP>:
warn dfs.datanode$dataxceiver: <IP:PORT>:got exception while serving blk_<ID> to /<IP>:
warn dfs.datanode$dataxceiver: <IP:PORT>:got exception while serving blk_<ID> to /<IP>:"""

# Create Gradio interface
with gr.Blocks(title="HDFS Incident Triage", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔧 HDFS Incident Triage System
    
    Automatically classify incident severity, identify root causes, and recommend actions using a fine-tuned FLAN-T5 model.
    
    **Model**: FLAN-T5-small fine-tuned on HDFS logs  
    **Accuracy**: 89% severity, 78% cause identification  
    **Features**: Structured JSON output, robust repair pipeline, heuristic overrides
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            incident_input = gr.Textbox(
                label="Incident Logs",
                placeholder="Paste HDFS log lines here (normalized format)...",
                lines=10,
                value=EXAMPLE_1
            )
            
            with gr.Row():
                submit_btn = gr.Button("🔍 Triage Incident", variant="primary", size="lg")
                clear_btn = gr.ClearButton(variant="secondary")
            
            gr.Examples(
                examples=[EXAMPLE_1, EXAMPLE_2],
                inputs=incident_input,
                label="Example Incidents"
            )
        
        with gr.Column(scale=2):
            severity_output = gr.Textbox(label="📊 Severity", interactive=False)
            cause_output = gr.Textbox(label="🎯 Likely Cause", interactive=False)
            action_output = gr.Textbox(label="💡 Recommended Action", lines=3, interactive=False)
            confidence_output = gr.Textbox(label="✅ Confidence", interactive=False)
    
    with gr.Accordion("🔍 View Raw Model Output", open=False):
        raw_output = gr.Textbox(label="Raw Model Output", lines=5, interactive=False)
    
    gr.Markdown("""
    ---
    ### 📌 How It Works
    
    1. **Structured Completion**: Model fills JSON template slots instead of free-form generation
    2. **Normalization**: Fixes smart quotes, parentheses, malformed syntax
    3. **Intelligent Repair**: Extracts values using regex, applies sensible defaults
    4. **Heuristic Override**: Domain rules correct known patterns (e.g., "exception while serving" → SEV-1)
    
    ### 🎯 Model Performance
    - **Severity Classification**: 89% accuracy
    - **Cause Identification**: 78% accuracy  
    - **Valid JSON Output**: 100% (guaranteed)
    
    ### 📚 Learn More
    - [GitHub Repository](https://github.com/PragatiAN1109/incident-triage-llm)
    - [Technical Report](https://github.com/PragatiAN1109/incident-triage-llm/blob/main/docs/technical_report.md)
    """)
    
    # Connect button
    submit_btn.click(
        fn=triage_incident,
        inputs=incident_input,
        outputs=[severity_output, cause_output, action_output, raw_output, confidence_output]
    )
    
    clear_btn.add([incident_input, severity_output, cause_output, action_output, raw_output, confidence_output])

# Launch
if __name__ == "__main__":
    demo.launch()
