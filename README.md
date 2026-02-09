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
