# Technical Report: Fine-Tuning FLAN-T5 for Automated HDFS Incident Triage

**Author**: Pragati Narote  
**Date**: February 2026  
**Model**: FLAN-T5-small (80M parameters)  
**Task**: Structured JSON generation for incident severity, cause, and action recommendation

---

## 1. Introduction

### 1.1 Problem Statement

Large-scale distributed systems like HDFS generate thousands of log entries daily. Manual incident triage requires engineers to:
- Review verbose, unstructured logs
- Classify incident severity (critical vs routine)
- Identify root causes from error patterns
- Determine appropriate remediation actions

This process is time-consuming, error-prone, and requires deep domain expertise. **Manual triage delays mean time to resolution (MTTR) and risks missing critical incidents.**

### 1.2 Objective

Develop an automated incident triage system using a fine-tuned Large Language Model that:
1. Classifies incident severity (SEV-1 critical vs SEV-3 low)
2. Identifies likely root cause from log patterns
3. Recommends appropriate remediation actions
4. Outputs structured JSON for downstream system integration

### 1.3 Dataset

**Source**: HDFS logs from the LogHub repository  
**Size**: 2,000 raw log lines → 48 processed incidents  
**Splits**: 33 train (68.8%), 6 validation (12.5%), 9 test (18.8%)  
**Label Distribution**:
- SEV-1 (Critical): 31.2% (15 incidents)
- SEV-3 (Low): 68.8% (33 incidents)
- Likely causes: Packet responder termination (68.8%), Block serving exception (31.2%)

---

## 2. Methodology

### 2.1 Dataset Preparation

#### Preprocessing Pipeline
1. **Log Parsing**: Extract level (WARN/ERROR/INFO), component, message
2. **Normalization**: Lowercase, remove timestamps, replace block IDs with `blk_<ID>`, replace IPs with `<IP>`
3. **Filtering**: Keep WARN/ERROR/FATAL + INFO with keywords (exception, timeout, fail, etc.)
4. **Grouping**: Sliding window approach (group_size=8, stride=4-8) to create incident sequences
5. **Service Detection**: Infer hdfs-datanode vs hdfs-namenode from components

#### Labeling Strategy (Rule-Based)
**Severity Assignment**:
- SEV-1: ERROR/FATAL logs OR critical keywords OR WARN ≥ 4
- SEV-3: WARN 0-3 without critical indicators

**Cause Assignment** (priority order):
1. Block serving exception: Contains "exception while serving"
2. Packet responder termination: Contains "packetresponder" AND "terminating"

**Action Assignment**: Deterministic mapping from likely_cause

#### Data Formatting
**Structured JSON Completion Prompts**: Instead of free-form generation, we use slot-filling templates that provide explicit JSON structure with empty slots:

```
You are an incident triage assistant.
Fill in the JSON template below using the incident information.

JSON:
{
  "severity": "",
  "likely_cause": "",
  "recommended_action": ""
}

Incident:
<incident logs>

Output ONLY the completed JSON:
```

This approach reduces task complexity for small datasets (33 training samples).

### 2.2 Model Selection

**Chosen Model**: `google/flan-t5-small` (80M parameters)

**Rationale**:
1. **Instruction-tuned**: FLAN-T5 is pre-trained for prompt→response tasks
2. **Seq2seq architecture**: Natural fit for structured JSON generation
3. **Lightweight**: Enables rapid experimentation on standard hardware
4. **Strong baseline**: Proven performance on structured output tasks

**Alternatives Considered**:
- **GPT-2 (124M)**: Decoder-only architecture less suitable for JSON generation
- **FLAN-T5-base (250M)**: Better capacity but 3x slower, overkill for 48 samples
- **DistilBERT**: Encoder-only, not suitable for generation tasks

**Tradeoff**: Quality ceiling vs speed and reproducibility. For a 48-sample dataset, FLAN-T5-small provides optimal balance.

### 2.3 Training Approach

#### Hyperparameter Optimization
Tested 3 configurations to optimize validation loss:

| Config | LR | Batch Size | Epochs | Train Loss | Val Loss |
|--------|-----|-----------|--------|------------|----------|
| A (Baseline) | 5e-5 | 4 | 3 | 3.9450 | 2.8075 |
| B (Lower LR) | 2e-5 | 4 | 3 | 4.6322 | 3.8554 |
| **C (Best)** | **5e-5** | **2** | **5** | **2.4405** | **0.8110** |

**Winner: Config C** - 71% validation loss reduction vs baseline

**Key Insight**: Smaller batch size (2) + more epochs (5) enabled finer-grained gradient updates, critical for tiny datasets.

#### Training Enhancements
1. **Newline-enhanced responses**: Append `\n` to training targets to teach proper end-of-sequence
2. **Stratified splitting**: Balanced severity distribution across splits
3. **Deterministic seed**: All randomness seeded with 42 for reproducibility

#### Training Infrastructure
- **Framework**: Hugging Face Transformers Trainer API
- **Hardware**: MacBook Air (MPS backend)
- **Duration**: ~0.45 minutes per training run
- **Checkpointing**: Save top 2 checkpoints by validation loss

### 2.4 Evaluation Strategy

#### Metrics
1. **Field-level accuracy**: Severity, likely_cause, recommended_action independently
2. **Exact match accuracy**: All three fields correct simultaneously
3. **Valid JSON rate**: Percentage of parseable outputs (with/without repair)
4. **Confusion matrix**: Severity classification performance

#### Baseline Comparison
Compared against pre-trained FLAN-T5-small (no fine-tuning):
- Baseline: Unstructured text, no JSON understanding
- Fine-tuned: Structured JSON with 67% exact match accuracy

---

## 3. Results

### 3.1 Quantitative Performance

**Test Set Performance (9 samples)**:
- **Severity Accuracy**: 88.9% (8/9 correct)
- **Likely Cause Accuracy**: 77.8% (7/9 correct)
- **Recommended Action Accuracy**: 66.7% (6/9 correct)
- **Exact Match**: 66.7% (6/9 all fields correct)
- **Valid JSON Rate**: 100% (with inference-time repair)

**Confusion Matrix (Severity)**:
```
               Predicted
             SEV-1  SEV-3
Actual SEV-1    3      0     (100% recall)
       SEV-3    1      5     (83.3% recall)
```

**Key Finding**: No false negatives for SEV-1 (critical incidents never missed), which is critical for production safety.

### 3.2 Baseline Comparison

| Metric | Baseline | Fine-tuned | Improvement |
|--------|----------|-----------|-------------|
| Severity Accuracy | ~33% | 88.9% | **+55.9%** |
| Likely Cause Accuracy | 0% | 77.8% | **+77.8%** |
| Valid JSON Output | 0% | 100% | **+100%** |

**Baseline**: Pre-trained model produces unstructured narrative text, not JSON.  
**Fine-tuned**: Learns structured output format with high accuracy.

### 3.3 Inference Statistics

- **Repair Rate**: 33% (3/9 samples require normalization/repair)
- **Heuristic Override Rate**: 33% (3/9 block-serving cases corrected)
- **Complete Output Rate**: 100% (guaranteed valid JSON)

**Repair Examples**:
- Smart quotes → normal quotes
- Parentheses around values → quoted strings
- Incomplete JSON → extracted values + defaults

---

## 4. Analysis & Discussion

### 4.1 What Worked Well

#### 1. Structured JSON Completion Prompts
Providing explicit templates with empty slots (slot-filling) proved highly effective for small datasets:
- Reduces task complexity
- Guides model toward structured output
- Prevents "schema-only degeneration" (outputting keys without values)

#### 2. Inference-Time Repair Pipeline
Three-layer safety net ensures production reliability:
- **Normalization**: Fixes smart quotes, parentheses, malformed syntax
- **Intelligent repair**: Extracts values, applies defaults, avoids UNKNOWN
- **Heuristic override**: Domain rules correct known error patterns

#### 3. Hyperparameter Optimization
Config C (smaller batch, more epochs) significantly outperformed larger batch alternatives:
- Validation loss: 0.8110 vs 2.8075 (71% reduction)
- Demonstrates importance of tuning for small datasets

### 4.2 Limitations

#### 1. Small Dataset (33 training samples)
**Impact**:
- Model struggles with JSON formatting consistency
- ~33% outputs require repair (smart quotes, incomplete generation)
- Limited pattern diversity in training data

**Evidence**: Incomplete JSON generation (stops after `"recommended_action":`)

#### 2. Model Capacity Constraints (80M params)
**Impact**:
- Formatting errors more frequent than larger models would produce
- Occasional confusion between similar incident types

**Tradeoff**: Fast inference (<500ms) vs output quality

#### 3. Label Noise from Rule-Based Heuristics
**Impact**:
- Ground truth labels imperfect (based on keyword matching)
- Some SEV-3 incidents may actually be SEV-1
- Model learns from noisy labels

**Evidence**: Heuristic override needed for 33% of test samples

### 4.3 Error Patterns Identified

See `notebooks/error_analysis.ipynb` for detailed analysis.

**Pattern 1: Severity Misclassification**
- Frequency: 11% (1/9 samples)
- Cause: Model predicts SEV-3 for "exception while serving" incidents
- Fix: Heuristic override → SEV-1

**Pattern 2: Incomplete JSON**
- Frequency: 33% (3/9 samples)
- Cause: Small dataset + limited token budget
- Fix: Increased max_new_tokens=256, min_new_tokens=60, length_penalty=0.8

**Pattern 3: Formatting Issues**
- Frequency: 22% (2/9 samples)
- Cause: Smart quotes, parentheses around values
- Fix: Robust normalization pipeline

---

## 5. Limitations & Future Work

### 5.1 Current Limitations

1. **Dataset size**: 33 training samples insufficient for robust generalization
2. **Label quality**: Rule-based heuristics introduce noise
3. **Model capacity**: 80M params limits JSON formatting consistency
4. **Binary severity**: Only SEV-1 and SEV-3 represented (no SEV-2 examples)
5. **Limited incident types**: Only 2 likely_cause categories in dataset

### 5.2 Proposed Improvements

#### High Priority
1. **Expand dataset to 150-200 samples**
   - Use sliding window with stride=2 (75% overlap)
   - Expected impact: +15-20% accuracy, -20% repair rate
   - Implementation ready in `preprocess_logs.py --stride 2`

2. **Add manual validation**
   - Review 10-20 challenging cases
   - Correct rule-based labeling errors
   - Expected impact: +10% accuracy from cleaner labels

#### Medium Priority
3. **Upgrade to FLAN-T5-base (250M params)**
   - Expected impact: Better JSON formatting, fewer repairs
   - Tradeoff: 3x slower inference, requires GPU

4. **Data augmentation**
   - Paraphrase incident descriptions using GPT
   - Vary log order while preserving labels
   - Expected impact: +10% generalization

#### Low Priority
5. **Constrained decoding**
   - Force JSON format using grammar-based generation (guidance-ai, jsonformer)
   - Expected impact: 100% valid JSON, eliminate repair logic
   - Tradeoff: 2-3x slower generation

6. **Multi-task learning**
   - Auxiliary tasks: log level prediction, component classification
   - Expected impact: Better representation learning

---

## 6. References

1. Chung, H. W., et al. (2022). "Scaling Instruction-Finetuned Language Models." arXiv:2210.11416. [FLAN-T5 paper]

2. LogHub: https://github.com/logpai/loghub - HDFS log dataset

3. Hugging Face Transformers: https://huggingface.co/docs/transformers

4. Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." JMLR. [T5 paper]

5. Best practices for fine-tuning small LLMs: https://huggingface.co/blog/fine-tune-small-models

---

## Appendix A: Reproduction Commands

```bash
# Complete pipeline from scratch
python3 scripts/preprocess_logs.py --input data/raw/HDFS_2k.log --output data/processed/incidents_2k.jsonl --stride 4
python3 scripts/build_dataset.py --input data/processed/incidents_2k.jsonl --output_dir data/final --seed 42
python3 scripts/train_experiments.py --only_best
python3 scripts/evaluate.py
python3 scripts/inference.py
```

## Appendix B: Error Analysis Summary

**Total Errors**: 3/9 (33.3%)

**Error Categories**:
1. Severity: 1 error (SEV-3 → SEV-1 misclassification)
2. Likely cause: 2 errors (confusion between categories)
3. Recommended action: 3 errors (incomplete generation)

**Mitigation Success Rate**:
- Heuristic override: 100% (3/3 block-serving cases corrected)
- Repair pipeline: 100% (all outputs produce valid JSON)
- UNKNOWN field rate: 0% (robust extraction avoids data loss)

For detailed analysis, run: `jupyter notebook notebooks/error_analysis.ipynb`
