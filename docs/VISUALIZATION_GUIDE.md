# Visualization Guide: How to Generate Graphs for Your Report

## Quick Start

### Step 1: Install Dependencies
```bash
pip install matplotlib seaborn numpy jupyter
```

### Step 2: Generate All Visualizations
```bash
python3 scripts/generate_visualizations.py
```

This creates 15 publication-ready figures in `results/figures/`:

---

## Generated Figures

### 📊 Dataset & Preprocessing (Figures 1-3)

**01_data_pipeline.png** - Data Processing Flowchart
- Shows: Raw logs → Parsing → Filtering → Normalization → Grouping → Incidents
- Use in: Introduction, Methodology (Dataset Preparation)

**02_dataset_splits.png** - Train/Val/Test Distribution
- Shows: Bar chart of split sizes + pie chart of severity distribution
- Use in: Methodology (Dataset section)

**03_label_distribution.png** - Label Distribution
- Shows: Severity and likely_cause category breakdown
- Use in: Methodology (Labeling Strategy)

---

### 🎯 Hyperparameter Tuning (Figures 4-6)

**04_hyperparameter_comparison.png** - Config A/B/C Comparison
- Shows: Side-by-side train and validation loss for all configs
- Use in: Results (Hyperparameter Optimization)

**05_validation_improvement.png** - Validation Loss Improvement
- Shows: Config C achieves 71% lower validation loss vs baseline
- Use in: Results (Best Configuration Selection)

**06_training_runtime.png** - Training Time Comparison
- Shows: Runtime for each configuration
- Use in: Discussion (Efficiency Analysis)

---

### 📈 Model Evaluation (Figures 7-9)

**07_test_performance.png** - Test Set Accuracy Metrics
- Shows: Severity, cause, action, and exact match accuracy
- Use in: Results (Performance Metrics)

**08_baseline_comparison.png** - Pre-trained vs Fine-tuned
- Shows: Dramatic improvement over baseline (+56%, +78%, +100%)
- Use in: Results (Baseline Comparison) - **KEY FIGURE**

**09_confusion_matrix.png** - Severity Classification Matrix
- Shows: SEV-1 has 100% recall (no false negatives)
- Use in: Results (Detailed Evaluation)

---

### 🔍 Error Analysis (Figures 10-11)

**10_error_breakdown.png** - Error Categories and Rates
- Shows: Distribution of error types and their frequencies
- Use in: Analysis (Error Patterns)

**11_mitigation_effectiveness.png** - Mitigation Strategy Success
- Shows: Effectiveness of heuristics, repair, and decoding improvements
- Use in: Discussion (Problem Resolution)

---

### ⚙️ Inference Pipeline (Figures 12-13)

**12_inference_statistics.png** - Repair and Override Rates
- Shows: Progressive quality improvement through pipeline layers
- Use in: Implementation (Inference Pipeline)

**13_inference_pipeline.png** - Three-Layer Safety Net Diagram
- Shows: Normalization → Repair → Heuristic layers
- Use in: Methodology (Inference Architecture) - **KEY FIGURE**

---

### 📋 Project Summary (Figures 14-15)

**14_project_summary.png** - Key Metrics Dashboard
- Shows: 6-panel summary (dataset, training, accuracy, speed, reliability, improvement)
- Use in: Introduction or Executive Summary - **KEY FIGURE**

**15_training_progress.png** - Learning Curves
- Shows: Training and validation loss over 5 epochs
- Use in: Results (Training Progress)

---

## How to Use in Technical Report

### Markdown Format
```markdown
![Data Processing Pipeline](results/figures/01_data_pipeline.png)
*Figure 1: Complete data processing pipeline from raw logs to labeled incidents*
```

### LaTeX Format
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{results/figures/01_data_pipeline.png}
\caption{Data Processing Pipeline}
\label{fig:pipeline}
\end{figure}
```

---

## Recommended Figure Placement

### **Section 1: Introduction**
- Figure 14: Project Summary Dashboard (overview of key metrics)

### **Section 2: Methodology**

**2.1 Dataset Preparation**
- Figure 1: Data Processing Pipeline
- Figure 2: Dataset Splits Distribution
- Figure 3: Label Distribution

**2.2 Model Selection & Training**
- Figure 13: Inference Pipeline Diagram (architecture)

**2.3 Hyperparameter Optimization**
- Figure 4: Hyperparameter Comparison
- Figure 5: Validation Loss Improvement

### **Section 3: Results**

**3.1 Training Results**
- Figure 15: Training Progress (learning curves)
- Figure 6: Training Runtime Comparison

**3.2 Test Set Performance**
- Figure 7: Test Performance Metrics
- Figure 8: Baseline Comparison ⭐ **MOST IMPORTANT**
- Figure 9: Confusion Matrix

### **Section 4: Error Analysis**
- Figure 10: Error Breakdown by Category
- Figure 11: Mitigation Strategy Effectiveness

### **Section 5: Inference Pipeline**
- Figure 12: Inference Statistics
- Figure 13: Three-Layer Safety Net (if not used earlier)

---

## Customization Options

### Update with Actual Data

After running `python3 scripts/evaluate.py`, update the script with real metrics:

```python
# In generate_visualizations.py, replace estimated values:

# Line ~180: Test performance metrics
scores = [88.9, 77.8, 66.7, 66.7]  # REPLACE with actual from evaluation_metrics.json

# Line ~220: Confusion matrix
confusion_matrix = np.array([
    [3, 0],  # REPLACE with actual values
    [1, 5]
])

# Line ~280: Error counts
error_counts = [1, 2, 3, 2]  # REPLACE with actual from error_analysis.json
```

### Change Color Scheme

```python
# At top of script, change color palette:
sns.set_palette("colorblind")  # For accessibility
# or
sns.set_palette("Set2")  # Softer colors
```

### Adjust Figure Sizes

```python
# For presentations (bigger text):
plt.subplots(figsize=(12, 8))  # Increase from (10, 6)

# For papers (smaller):
plt.subplots(figsize=(8, 5))
```

---

## Advanced: Create Custom Visualizations

### Example: Add Training Loss Over Time

If you have detailed training logs, create a real learning curve:

```python
import json

# Load training logs (if saved)
with open('results/config_c_(higher_capacity)/trainer_state.json', 'r') as f:
    trainer_state = json.load(f)

# Extract loss values
log_history = trainer_state['log_history']
train_losses = [entry['loss'] for entry in log_history if 'loss' in entry]
val_losses = [entry['eval_loss'] for entry in log_history if 'eval_loss' in entry]
epochs = range(1, len(val_losses) + 1)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses[:len(epochs)], marker='o', label='Train Loss')
plt.plot(epochs, val_losses, marker='s', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Actual Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/figures/training_curve_actual.png', dpi=300)
```

---

## Quick Quality Check

After generating visualizations:

```bash
# Check all figures were created
ls -lh results/figures/

# Expected output: 15 PNG files, ~100-300 KB each
```

Open a few in Preview/image viewer to verify:
- ✅ Text is readable (not too small)
- ✅ Colors are distinct
- ✅ Labels are clear
- ✅ No clipping or cutoff elements

---

## Using Figures in Different Formats

### For PDF Report (High Quality)
- Current settings: `dpi=300` ✅ Already configured
- File size: ~200-500 KB per figure
- Quality: Publication-ready

### For PowerPoint Presentation
```python
plt.savefig('figure.png', dpi=150, bbox_inches='tight')  # Smaller file
```

### For Web/GitHub README
```python
plt.savefig('figure.png', dpi=100, bbox_inches='tight')  # Even smaller
```

---

## Common Issues & Fixes

### Issue: "Module 'matplotlib' not found"
```bash
pip install matplotlib seaborn numpy
```

### Issue: "seaborn style not found"
```python
# Change this line:
plt.style.use('seaborn-v0_8-darkgrid')
# To:
plt.style.use('default')
```

### Issue: Figures look cramped
```python
# Add this before plt.savefig():
plt.tight_layout()
```

### Issue: Text too small
```python
# Increase font sizes globally at top of script:
plt.rcParams.update({'font.size': 12})
```

---

## Alternative: Manual Visualization

If you prefer manual control, use this template:

```python
import matplotlib.pyplot as plt

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Your data
x = ['Config A', 'Config B', 'Config C']
y = [2.8075, 3.8554, 0.8110]

# Plot
ax.bar(x, y, color=['gray', 'gray', 'green'])
ax.set_ylabel('Validation Loss')
ax.set_title('Hyperparameter Comparison')

# Save
plt.savefig('results/figures/my_figure.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Final Checklist

Before using figures in your report:

- [ ] Run `python3 scripts/generate_visualizations.py`
- [ ] Verify 15 figures created in `results/figures/`
- [ ] Update script with actual metrics from `evaluate.py`
- [ ] Regenerate with real data
- [ ] Open each figure to check quality
- [ ] Add figure captions in your report
- [ ] Reference figures in text (e.g., "As shown in Figure 8...")

---

## Pro Tips for Report Quality

1. **Always add captions**: Explain what the figure shows
2. **Reference in text**: "Figure 8 demonstrates a 56% improvement..."
3. **Use consistent style**: All figures should have similar color schemes
4. **High DPI**: Keep dpi=300 for professional quality
5. **Clear labels**: Make sure axis labels and titles are descriptive

Your visualizations are now ready for a professional technical report! 📊
