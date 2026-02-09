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

A sample of 50 processed incidents is available at:
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
