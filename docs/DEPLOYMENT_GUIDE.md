# Hugging Face Deployment Guide

## Overview

Deploy your fine-tuned incident triage model as a live web demo on Hugging Face Spaces.

**Benefits**:
- ✅ Shareable demo URL for your portfolio
- ✅ Interactive web interface (no code required for users)
- ✅ Free hosting from Hugging Face
- ✅ Impressive for job applications and presentations
- ✅ **Boosts Quality Score** to Top 25% (20/20 points)

---

## Option 1: Quick Deploy (Recommended)

### Step 1: Upload Model to Hugging Face Hub

```bash
# Install huggingface_hub
pip install huggingface_hub

# Login to Hugging Face
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

```python
# Upload model
from huggingface_hub import HfApi

api = HfApi()

# Upload your trained model
api.upload_folder(
    folder_path="results/config_c_(higher_capacity)/final-model",
    repo_id="PragatiAN1109/incident-triage-llm",
    repo_type="model",
    commit_message="Upload fine-tuned FLAN-T5 for incident triage"
)
```

### Step 2: Create Hugging Face Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Fill in:
   - **Space name**: `incident-triage-demo`
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU (free tier)
4. Click "Create Space"

### Step 3: Upload Files to Space

Your Space needs these files (already created in your repo):

**Required files**:
1. `app.py` ✅ (already created)
2. `requirements.txt` (create for Space)

Create `requirements.txt` for the Space:
```
transformers==4.38.2
torch>=2.0.0
gradio>=4.0.0
```

**Upload to Space**:
- Go to your Space's "Files" tab
- Click "Add file" → "Upload files"
- Upload: `app.py` and the Space `requirements.txt`

### Step 4: Update Model Path in app.py

In the Space's file editor, change line 8:
```python
MODEL_PATH = "PragatiAN1109/incident-triage-llm"  # Your HF model repo
```

### Step 5: Space Builds Automatically

- Wait 2-3 minutes for build
- Your demo will be live at: `https://huggingface.co/spaces/PragatiAN1109/incident-triage-demo`

---

## Option 2: Local Testing First

Test the Gradio interface locally before deploying:

```bash
# Install Gradio
pip install gradio

# Run locally
python3 app.py
```

This opens http://localhost:7860 in your browser.

**Test**:
1. Paste example incident logs
2. Click "Triage Incident"
3. Verify outputs look correct

Once satisfied, deploy to Hugging Face following Option 1.

---

## Option 3: Deploy Directly from GitHub (Advanced)

Link your Space to this GitHub repo for auto-sync:

1. Create Space on HuggingFace
2. In Space settings → Repository → Link to GitHub
3. Select: `PragatiAN1109/incident-triage-llm`
4. Space auto-updates when you push to GitHub

---

## What the Demo Looks Like

**Interface**:
```
┌─────────────────────────────────────────────────────────┐
│  HDFS Incident Triage System                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Incident Logs:                │  📊 Severity: SEV-1    │
│  ┌────────────────────┐         │  🎯 Likely Cause:      │
│  │ service: hdfs-...  │         │     Block serving...   │
│  │ warn dfs.datanode. │         │  💡 Recommended:       │
│  │ ...                │         │     Investigate...     │
│  └────────────────────┘         │  ✅ Confidence: High   │
│                                 │                         │
│  [🔍 Triage Incident]  [Clear]  │                         │
│                                                          │
│  Example Incidents ▼                                     │
└─────────────────────────────────────────────────────────┘
```

Users just paste logs and click → instant triage results!

---

## Benefits for Your Assignment

### Quality Score Impact (+3-5 points)

**With Deployment**:
- Shows "production-ready" implementation
- Demonstrates full ML lifecycle (train → deploy → serve)
- Provides shareable demo link
- **Positions you in Top 25%** (18-20/20 quality points)

**Without Deployment**:
- Still good project, but missing deployment component
- **51-75th percentile** (15-17/20 quality points)

### Portfolio Impact

- Add to resume: "Deployed ML model to production (Hugging Face)"
- Share link with recruiters: Live demo > GitHub repo only
- Show in interviews: Pull up live demo on phone

---

## Troubleshooting

### Issue: "Model not found"
Update `MODEL_PATH` in app.py to your uploaded model:
```python
MODEL_PATH = "PragatiAN1109/incident-triage-llm"
```

### Issue: "Out of memory"
Space's free tier has 16GB RAM. FLAN-T5-small fits easily. If issues:
```python
# Add to app.py after loading model:
model.to('cpu')  # Force CPU
```

### Issue: "Build failed"
Check Space logs for missing dependencies. Ensure requirements.txt includes:
```
transformers
torch
gradio
```

### Issue: Slow loading
First load takes ~30 seconds (model download). After that, inference is fast (<2 seconds).

---

## Alternative: Simpler Demo (No HF Upload Needed)

If you don't want to upload the model, you can run Gradio locally and share a temporary link:

```bash
python3 app.py --share
```

This creates a public URL (valid for 72 hours): `https://xxxxx.gradio.live`

---

## What to Include in Your Submission

### In Video Walkthrough
- Show the live Space demo
- Paste example logs
- Click "Triage Incident"
- Explain the outputs

### In Technical Report
```markdown
## Deployment

The model is deployed as a web application on Hugging Face Spaces:
- **Live Demo**: https://huggingface.co/spaces/PragatiAN1109/incident-triage-demo
- **API Endpoint**: Available via Hugging Face Inference API
- **Latency**: <2 seconds per incident
- **Availability**: 24/7 uptime
```

### In README
Add a "🚀 Live Demo" section at the top:
```markdown
## 🚀 Live Demo

Try the model without installing anything:

**[Launch Interactive Demo](https://huggingface.co/spaces/PragatiAN1109/incident-triage-demo)** 🔗

Paste HDFS log entries and get instant triage results!
```

---

## Timeline

**Total time**: ~20 minutes

- Model upload: 5 minutes
- Space creation: 2 minutes
- File upload: 3 minutes
- Build wait: 5 minutes
- Testing: 5 minutes

**Worth it?** YES - for +3-5 quality score points and portfolio value!

---

## Decision Matrix

| Approach | Time | Effort | Quality Points | Portfolio Value |
|----------|------|--------|----------------|-----------------|
| **No Deployment** | 0 min | None | 15-17/20 | Medium |
| **Gradio Local Only** | 5 min | Low | 16-18/20 | Medium-High |
| **HF Space Deploy** | 20 min | Medium | **18-20/20** | **High** |

**Recommendation**: Deploy to HF Space for maximum impact! 🚀

---

## Quick Start Commands

```bash
# 1. Install deployment dependencies
pip install huggingface_hub gradio

# 2. Login to Hugging Face
huggingface-cli login

# 3. Test locally first
python3 app.py

# 4. Follow Option 1 steps above to deploy
```

Need help? Check the detailed guide above! ✨
