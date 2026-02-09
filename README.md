# incident-triage-llm
Fine-tuning a Large Language Model for automated incident triage using system logs

## Data Cleaning & Preprocessing

The preprocessing pipeline transforms raw HDFS logs into structured incident records suitable for LLM fine-tuning. The script performs the following operations:

- **Parsing**: Extracts log level, component, and message from each log line
- **Normalization**: Converts text to lowercase, strips timestamps and numeric IDs, collapses whitespace, and replaces block IDs with `blk_<ID>` placeholder
- **Filtering**: Keeps WARN/ERROR/FATAL logs and INFO logs containing keywords like "retry", "timeout", "fail", "exception", "terminating", "unable", "refused", "exceeded", "corrupt", "denied", etc.
- **Grouping**: Groups consecutive log lines into incidents of a specified size (default: 8 lines per incident)
- **Service Detection**: Infers service type (hdfs-datanode, hdfs-namenode, or hdfs) from log content
- **Hashing**: Generates unique incident IDs using SHA-1 hash

### How to Run

**Dry run (preview without writing output):**
```bash
python3 scripts/preprocess_logs.py --dry_run
```

**Generate processed output:**
```bash
# Default: processes data/raw/HDFS_2k.log → data/processed/incidents.jsonl
python3 scripts/preprocess_logs.py

# Custom parameters
python3 scripts/preprocess_logs.py \
  --input data/raw/HDFS_2k.log \
  --output data/processed/my_incidents.jsonl \
  --group_size 8 \
  --max_incidents 500 \
  --seed 42
```

### Committed Sample

A sample of 48 processed incidents is available at:
```
data/processed/sample_incidents_50.jsonl
```

Each incident is stored as a JSON line with the following structure:
```json
{
  "incident_id": "321a4db2e756",
  "service": "hdfs-datanode",
  "lines": [...],
  "incident_text": "...",
  "stats": {"levels_count": {...}, "top_components": {...}}
}
```

**Example incident_text (normalized log lines):**
```
info dfs.datanode$packetresponder: packetresponder 1 for block blk_<ID> terminating
info dfs.datanode$packetresponder: packetresponder 0 for block blk_<ID> terminating
info dfs.datanode$packetresponder: packetresponder 2 for block blk_<ID> terminating
warn dfs.datanode$dataxceiver: <IP:PORT>:got exception while serving blk_<ID> to /<IP>:
```

## Dataset Formatting & Splitting

The dataset builder (`scripts/build_dataset.py`) transforms preprocessed incidents into a fine-tuning-ready format with rule-based labels.

### Labeling Strategy

Labels are assigned deterministically using keyword-based heuristics:

**Severity** (3 levels):
- **SEV-1 (Critical)**: Any ERROR/FATAL level OR critical keywords ("exception", "fatal", "error", "failed") OR phrase "exception while serving" OR WARN count ≥ 4
- **SEV-2 (High)**: WARN count 2-3
- **SEV-3 (Low)**: WARN count 0-1

**Likely Cause** (5 categories, strict priority order):
1. **Block serving exception**: Contains "exception while serving"
2. **Network connectivity issue**: Contains "timeout", "refused", "unreachable", or "disconnect"
3. **Service degradation**: Contains "slow", "lag", "backlog", or "thrott"
4. **Packet responder termination**: Contains both "packetresponder" AND "terminating"
5. **HDFS operational event**: Default category (no matches above)

**Recommended Action**: Mapped deterministically from the likely cause:
- Block serving exception → "Investigate DataNode block serving failures and check network connectivity between nodes"
- Packet responder termination → "Monitor DataNode for abnormal packet responder patterns; verify normal block replication"
- Network connectivity issue → "Check network configuration and firewall rules; verify DataNode-NameNode connectivity"
- Service degradation → "Review system resources (CPU, memory, disk I/O); check for performance bottlenecks"
- HDFS operational event → "Monitor cluster health metrics; no immediate action required unless pattern persists"

### Prompt/Response Format

Each training example contains:
- **prompt**: The `incident_text` field (service + log lines)
- **response**: A JSON string with three fields:
  ```json
  {
    "severity": "SEV-1",
    "likely_cause": "Block serving exception",
    "recommended_action": "Investigate DataNode block serving failures..."
  }
  ```

### Split Ratios and Reproducibility

The dataset is split with **stratification by severity** to ensure balanced representation:
- **Train**: 70%
- **Validation**: 15%
- **Test**: 15%

**Actual Counts**: The script prints exact counts when run. For the sample dataset (`sample_incidents_50.jsonl` with 48 complete incidents), the distribution is:
```
Total samples: 48

Train: 33 samples
  SEV-1: 10
  SEV-3: 23

Validation: 6 samples
  SEV-1: 2
  SEV-3: 4

Test: 9 samples
  SEV-1: 3
  SEV-3: 6
```

**Reproducibility**: All splits use a fixed random seed (default: 42) for deterministic results across runs.

### How to Build Dataset

```bash
# Default: reads from data/processed/sample_incidents_50.jsonl
python3 scripts/build_dataset.py

# Custom parameters
python3 scripts/build_dataset.py \
  --input data/processed/sample_incidents_50.jsonl \
  --output_dir data/final \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --seed 42
```

**Output files:**
- `data/final/train.jsonl`
- `data/final/val.jsonl`
- `data/final/test.jsonl`

Each file contains JSON lines with `prompt` and `response` fields ready for fine-tuning.

## Model Selection

### Chosen Model: `google/flan-t5-small`

**Why This Model Fits:**
- **Instruction-tuned**: FLAN-T5 is specifically fine-tuned for prompt→response tasks, making it ideal for structured triage output generation
- **Lightweight**: The "small" variant (80M parameters) enables fast experimentation, rapid iteration, and easy reproducibility on standard hardware
- **Sequence-to-sequence architecture**: Well-suited for generating structured JSON responses from log incident prompts
- **Strong baseline**: Despite its size, FLAN-T5-small provides a solid foundation for task-specific fine-tuning

**Tradeoffs:**
- **Quality ceiling**: Smaller models may cap maximum achievable quality compared to larger variants (base, large, xl, xxl)
- **Benefit**: Enables rapid prototyping, faster training cycles, and reproducible experiments without requiring extensive computational resources

### Model Setup

Use `scripts/model_setup.py` to load the base model and run baseline inference before fine-tuning:

```bash
python3 scripts/model_setup.py
```

This script demonstrates:
- Loading the pre-trained model and tokenizer
- Running baseline inference on a sample incident
- Comparing generated output to ground-truth labels

## Fine-Tuning Setup

### Training Pipeline

The fine-tuning pipeline (`scripts/train.py`) uses the **Hugging Face Transformers Trainer API** for clean, reproducible training.

**Important**: The Trainer relies on Accelerate for distributed training support. Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

**Training Process:**
1. Load `google/flan-t5-small` base model and tokenizer
2. Load train and validation datasets from `data/final/`
3. Tokenize inputs (prompts) and targets (JSON responses)
4. Configure training with optimized hyperparameters
5. Train using Seq2SeqTrainer with automatic checkpointing
6. Save best model based on validation loss

### Dataset Splits Used

- **Training**: `data/final/train.jsonl` (33 samples)
- **Validation**: `data/final/val.jsonl` (6 samples)
- **Test**: `data/final/test.jsonl` (9 samples - held out, not used during training)

The test set is **not tokenized or used during training** - it remains completely unseen for final evaluation.

### Hyperparameters

**Core Training Settings:**
- **Epochs**: 3
- **Batch size**: 4 (train and eval)
- **Learning rate**: 5e-5
- **Weight decay**: 0.01
- **Optimizer**: AdamW (default)

**Sequence Lengths:**
- **Input (prompt)**: max 512 tokens
- **Output (response)**: max 256 tokens

**Evaluation & Checkpointing:**
- Evaluation strategy: Every epoch
- Save strategy: Every epoch
- Logging: Every 50 steps
- Save only best 2 checkpoints (based on validation loss)
- Load best model at end: Yes

**Reproducibility:**
- Fixed random seed: 42
- Deterministic tokenization and data loading

### Why This Setup is Reliable

1. **Stratified Splitting**: Ensures balanced severity distribution across train/val/test
2. **Fixed Seed**: All random operations use seed=42 for reproducibility
3. **Validation-Based Selection**: Best model chosen by validation loss, not training loss
4. **Standard Trainer API**: Uses well-tested Hugging Face infrastructure
5. **Automatic Checkpointing**: Prevents loss of progress and enables recovery
6. **Held-Out Test Set**: True generalization measured on completely unseen data

### How to Run Training

**Install dependencies first:**
```bash
pip install -r requirements.txt
```

**Run fine-tuning:**
```bash
python3 scripts/train.py
```

**Output:**
- Training checkpoints: `results/checkpoint-*/`
- Final fine-tuned model: `results/final-model/`
- Training logs printed to console

**Expected training time**: ~5-10 minutes on CPU, ~1-2 minutes on GPU (for 48 samples, 3 epochs)

## Task 7: Hyperparameter Tuning & Model Selection

Multiple training configurations were evaluated to identify the best-performing model based on validation loss. Each configuration was trained independently using the same dataset, model architecture, and random seed to ensure fair comparison.

### Experiment Results

| Configuration | Learning Rate | Batch Size | Epochs | Train Loss | Validation Loss | Runtime (min) |
|---------------|---------------|------------|--------|------------|-----------------|---------------|
| Config A (Baseline) | 5e-5 | 4 | 3 | 3.9450 | 2.8075 | 0.26 |
| Config B (Lower LR) | 2e-5 | 4 | 3 | 4.6322 | 3.8554 | 0.24 |
| Config C (Higher Capacity) | 5e-5 | 2 | 5 | 2.4405 | 0.8110 | 0.45 |

### Best Configuration Selection

**Selected: Config C (Higher Capacity)**

Config C was chosen as the optimal configuration due to achieving the **lowest validation loss (0.8110)**, which indicates superior generalization performance. While this configuration requires 73% longer training time (0.45 min vs 0.26 min for the baseline), the substantial improvement in validation loss (71% reduction from baseline) justifies the additional computational cost. The smaller batch size (2) combined with extended training (5 epochs) enabled finer-grained gradient updates and more thorough optimization.

### Reproducibility Note

All experiments used a fixed random seed (42) for deterministic data splitting, tokenization, and model initialization. Given the relatively small dataset size (33 training samples), individual run results may exhibit minor variance due to numerical precision and hardware differences. However, the performance trends across configurations remain consistent, with Config C consistently outperforming the alternatives in validation loss.

## Task 8: Final Evaluation & Inference

The inference script (`scripts/inference.py`) demonstrates the fine-tuned model's performance on completely unseen test data. The script automatically loads the latest checkpoint from the best-performing configuration (Config C) and applies decoding constraints to ensure stable, structured outputs.

### What This Script Demonstrates

The script loads the best-performing model (Config C from Task 7) and runs inference on held-out test samples to provide a qualitative evaluation of the model's capabilities:

- **Incident parsing**: Understanding normalized HDFS log patterns
- **Severity classification**: Categorizing incidents as SEV-1, SEV-2, or SEV-3
- **Cause identification**: Identifying likely root causes from log signatures
- **Action generation**: Producing actionable remediation recommendations

**Automatic Checkpoint Loading**: The inference script automatically detects and loads the latest checkpoint under `results/config_c_(higher_capacity)` by finding the checkpoint with the highest step number (e.g., `checkpoint-85`).

**Decoding Constraints**: To ensure stable structured outputs, the script uses:
- Instruction wrapping to guide JSON generation
- Repetition penalty (1.2) to reduce redundant text
- N-gram blocking (size 3) to prevent repeating phrases
- Beam search (4 beams) for quality
- Reduced max tokens (128) for concise responses
- Format validation to check JSON structure and severity values

### How to Run

**First, run the training experiments to generate models:**
```bash
python3 scripts/train_experiments.py
```

**Then run inference on the best model:**
```bash
python3 scripts/inference.py
```

**Optional arguments:**
```bash
# Custom model path
python3 scripts/inference.py --model_path results/config_a_(baseline)

# More test samples
python3 scripts/inference.py --num_samples 5
```

**Output**: The script displays test samples with:
- Input incident logs (truncated for readability)
- Model-generated triage response with format validation
- Ground-truth reference response

This provides a side-by-side comparison to assess the model's learned behavior on unseen data.
