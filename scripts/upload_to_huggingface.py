#!/usr/bin/env python3
"""
Upload fine-tuned model to Hugging Face Hub.
Run this after training to make model available for Space deployment.
"""

from huggingface_hub import HfApi, create_repo
from pathlib import Path

# Configuration
MODEL_PATH = "results/config_c_(higher_capacity)/final-model"
REPO_ID = "PragatiAN1109/incident-triage-flan-t5"  # Change to your username

print("="*70)
print("UPLOADING MODEL TO HUGGING FACE HUB")
print("="*70)

# Verify model exists
model_path = Path(MODEL_PATH)
if not model_path.exists():
    print(f"\n❌ Error: Model not found at {MODEL_PATH}")
    print("Please run training first: python3 scripts/train_experiments.py --only_best")
    exit(1)

if not (model_path / "config.json").exists():
    print(f"\n❌ Error: config.json not found in {MODEL_PATH}")
    exit(1)

print(f"\n✓ Model found at: {MODEL_PATH}")
print(f"✓ Uploading to: {REPO_ID}")

# Initialize API
api = HfApi()

# Create repository (if doesn't exist)
try:
    print(f"\nCreating repository: {REPO_ID}")
    create_repo(
        repo_id=REPO_ID,
        repo_type="model",
        exist_ok=True,
        private=False
    )
    print(f"✓ Repository created/verified")
except Exception as e:
    print(f"Note: {e}")

# Upload model files
print(f"\nUploading model files...")
try:
    api.upload_folder(
        folder_path=MODEL_PATH,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload fine-tuned FLAN-T5 for HDFS incident triage"
    )
    print(f"\n✅ SUCCESS! Model uploaded to: https://huggingface.co/{REPO_ID}")
    print(f"\n📝 Next steps:")
    print(f"1. Visit: https://huggingface.co/{REPO_ID}")
    print(f"2. Add model card with description")
    print(f"3. Create Hugging Face Space and use this model")
    
except Exception as e:
    print(f"\n❌ Upload failed: {e}")
    print(f"\nTroubleshooting:")
    print(f"1. Make sure you ran: huggingface-cli login")
    print(f"2. Check your token has WRITE permission")
    print(f"3. Try again or upload manually via website")
    exit(1)
