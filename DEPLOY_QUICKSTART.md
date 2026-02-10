# Hugging Face Deployment - Quick Start

## 5 Simple Steps (20 minutes total)

### STEP 1: Setup (3 min)
```bash
pip install huggingface_hub gradio
huggingface-cli login
# Paste your token from https://huggingface.co/settings/tokens
```

### STEP 2: Upload Model (5 min)
```bash
python3 scripts/upload_to_huggingface.py
```

### STEP 3: Create Space (3 min)
- Go to https://huggingface.co/new-space
- Name: incident-triage-demo
- SDK: Gradio
- Click Create

### STEP 4: Upload Files (5 min)
In Space, upload:
1. app.py (from your repo)
2. requirements.txt with:
   ```
   transformers==4.38.2
   torch>=2.0.0
   gradio>=4.0.0
   ```

### STEP 5: Update Model Path (1 min)
In app.py line 7, change to:
```python
MODEL_PATH = "PragatiAN1109/incident-triage-flan-t5"
```

## Done!
Live demo at: https://huggingface.co/spaces/PragatiAN1109/incident-triage-demo

## Impact
- +3 quality score points
- Live portfolio piece
- Top 25% guaranteed
