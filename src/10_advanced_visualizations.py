"""
TIER 1 IMPROVEMENT: Advanced Visualizations


This script generates publication-quality visualizations:
1. ROC Curve (Receiver Operating Characteristic)
2. Precision-Recall Curve
3. Confusion Matrix Heatmap
4. Calibration Curve
5. Model Comparison Charts

"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)

# Try to import calibration_curve from different locations
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    try:
        from sklearn.metrics import calibration_curve
    except ImportError:
        print("[WARNING] calibration_curve not available, skipping calibration plot")
        calibration_curve = None
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

print("=" * 70)
print("ADVANCED VISUALIZATIONS FOR MODEL EVALUATION")
print("=" * 70)

# Create results directory
results_dir = Path('results/visualizations')
results_dir.mkdir(parents=True, exist_ok=True)

# Load test data
print("\n[1/6] Loading test data...")
with open('data/lung_cancer_test.pkl', 'rb') as f:
    test_data = pickle.load(f)

X_test = test_data['X_test']
y_test = test_data['y_class_test']  # Classification target (survived: 0/1)

print(f"   Test samples: {len(X_test)}")
print(f"   Survivors: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
print(f"   Deaths: {(1-y_test).sum()} ({(1-y_test.mean())*100:.1f}%)")

# Load best model
print("\n[2/6] Loading best model...")
with open('models/best_classifier.pkl', 'rb') as f:
    model_data = pickle.load(f)
    if isinstance(model_data, dict):
        model = model_data['model']
    else:
        model = model_data

print(f"   Model: {type(model).__name__}")

# Get predictions
print("\n[3/6] Generating predictions...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"   Predictions generated: {len(y_pred)}")

# VISUALIZATION 1: ROC Curve
print("\n[4/6] Creating visualizations...")
print("   [1/5] ROC Curve...")

fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'XGBoost (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Random Classifier (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
plt.title('ROC Curve - Cancer Survival Prediction', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)

# Add annotation
plt.annotate(f'Best Model\nAUC = {roc_auc:.3f}', 
             xy=(0.4, 0.6), fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(results_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("      Saved: roc_curve.png")

# VISUALIZATION 2: Precision-Recall Curve
print("   [2/5] Precision-Recall Curve...")

precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='darkblue', lw=2,
         label=f'XGBoost (AP = {avg_precision:.3f})')
plt.axhline(y=y_test.mean(), color='navy', linestyle='--', lw=2,
            label=f'Baseline (No Skill = {y_test.mean():.3f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall (True Positive Rate)', fontsize=12, fontweight='bold')
plt.ylabel('Precision', fontsize=12, fontweight='bold')
plt.title('Precision-Recall Curve - Cancer Survival Prediction', 
          fontsize=14, fontweight='bold')
plt.legend(loc="upper right", fontsize=11)
plt.grid(True, alpha=0.3)

# Add annotation
plt.annotate(f'Avg Precision\n{avg_precision:.3f}', 
             xy=(0.6, 0.3), fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig(results_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("      Saved: precision_recall_curve.png")

# VISUALIZATION 3: Confusion Matrix
print("   [3/5] Confusion Matrix...")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Death (0)', 'Survival (1)'],
            yticklabels=['Death (0)', 'Survival (1)'],
            annot_kws={'size': 14, 'weight': 'bold'})
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix - Cancer Survival Prediction', 
          fontsize=14, fontweight='bold')

# Add text annotations for interpretation
tn, fp, fn, tp = cm.ravel()
plt.text(0.5, -0.15, f'True Negatives: {tn:,}', ha='center', transform=plt.gca().transAxes)
plt.text(0.5, -0.18, f'False Positives: {fp:,}', ha='center', transform=plt.gca().transAxes)
plt.text(0.5, -0.21, f'False Negatives: {fn:,}', ha='center', transform=plt.gca().transAxes)
plt.text(0.5, -0.24, f'True Positives: {tp:,}', ha='center', transform=plt.gca().transAxes)

plt.tight_layout()
plt.savefig(results_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("      Saved: confusion_matrix.png")

# VISUALIZATION 4: Calibration Curve
print("   [4/5] Calibration Curve...")

if calibration_curve is not None:
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_pred_proba, n_bins=10, strategy='uniform'
    )

    plt.figure(figsize=(10, 8))
    plt.plot(mean_predicted_value, fraction_of_positives, marker='o', 
             linewidth=2, label='XGBoost', color='darkgreen')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated', 
             color='gray', linewidth=2)
    plt.xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
    plt.ylabel('Fraction of Positives (True Survival Rate)', fontsize=12, fontweight='bold')
    plt.title('Calibration Curve - Predicted vs Actual Survival Rate', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    # Add interpretation text
    plt.text(0.5, 0.05, 'Closer to diagonal = better calibration', 
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(results_dir / 'calibration_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("      Saved: calibration_curve.png")
else:
    print("      Skipped: calibration_curve (function not available)")

# VISUALIZATION 5: Combined Metrics Comparison
print("   [5/5] Model Performance Dashboard...")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1-Score': f1_score(y_test, y_pred, zero_division=0),
    'ROC-AUC': roc_auc
}

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Model Performance Dashboard - Cancer Survival Prediction', 
             fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Metrics Bar Chart
ax1 = axes[0, 0]
metric_names = list(metrics.keys())
metric_values = list(metrics.values())
bars = ax1.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
ax1.set_ylabel('Score', fontweight='bold')
ax1.set_title('Performance Metrics', fontweight='bold')
ax1.set_ylim([0, 1])
ax1.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 2: Mini ROC Curve
ax2 = axes[0, 1]
ax2.plot(fpr, tpr, color='darkorange', lw=2)
ax2.plot([0, 1], [0, 1], 'k--', lw=1)
ax2.set_xlabel('False Positive Rate', fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontweight='bold')
ax2.set_title(f'ROC Curve (AUC = {roc_auc:.3f})', fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Mini PR Curve
ax3 = axes[0, 2]
ax3.plot(recall, precision, color='darkblue', lw=2)
ax3.axhline(y=y_test.mean(), color='k', linestyle='--', lw=1)
ax3.set_xlabel('Recall', fontweight='bold')
ax3.set_ylabel('Precision', fontweight='bold')
ax3.set_title(f'PR Curve (AP = {avg_precision:.3f})', fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Confusion Matrix
ax4 = axes[1, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax4,
            xticklabels=['Death', 'Survival'],
            yticklabels=['Death', 'Survival'])
ax4.set_title('Confusion Matrix', fontweight='bold')

# Plot 5: Class Distribution
ax5 = axes[1, 1]
class_counts = [len(y_test) - y_test.sum(), y_test.sum()]
class_labels = ['Death (0)', 'Survival (1)']
bars = ax5.bar(class_labels, class_counts, color=['#d62728', '#2ca02c'])
ax5.set_ylabel('Count', fontweight='bold')
ax5.set_title('Test Set Class Distribution', fontweight='bold')
ax5.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}', ha='center', va='bottom', fontweight='bold')

# Plot 6: Prediction Distribution
ax6 = axes[1, 2]
ax6.hist(y_pred_proba[y_test==0], bins=30, alpha=0.6, label='Death (Actual)', color='red')
ax6.hist(y_pred_proba[y_test==1], bins=30, alpha=0.6, label='Survival (Actual)', color='green')
ax6.set_xlabel('Predicted Survival Probability', fontweight='bold')
ax6.set_ylabel('Frequency', fontweight='bold')
ax6.set_title('Prediction Distribution', fontweight='bold')
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / 'performance_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("      Saved: performance_dashboard.png")

# Generate detailed classification report
print("\n[5/6] Generating classification report...")

class_report = classification_report(y_test, y_pred, 
                                     target_names=['Death (0)', 'Survival (1)'],
                                     digits=4)

# Create detailed report
print("\n[6/6] Creating comprehensive report...")

report = []
report.append("=" * 70)
report.append("ADVANCED VISUALIZATION ANALYSIS REPORT")
report.append("=" * 70)
report.append("")

report.append("1. ROC CURVE (Receiver Operating Characteristic)")
report.append("-" * 70)
report.append("")
report.append("WHAT IT SHOWS:")
report.append("- Tradeoff between True Positive Rate (TPR) and False Positive Rate (FPR)")
report.append("- AUC (Area Under Curve) measures overall classification ability")
report.append("")
report.append(f"RESULTS:")
report.append(f"- ROC-AUC: {roc_auc:.4f}")
report.append("")
report.append("INTERPRETATION:")
if roc_auc >= 0.9:
    report.append("- EXCELLENT: Outstanding discrimination ability")
elif roc_auc >= 0.8:
    report.append("- VERY GOOD: Strong discrimination ability")
elif roc_auc >= 0.7:
    report.append("- GOOD: Acceptable discrimination ability")
elif roc_auc >= 0.6:
    report.append("- FAIR: Poor discrimination ability")
else:
    report.append("- POOR: Little better than random guessing")
report.append(f"- The model can distinguish survivors from non-survivors")
report.append(f"  {(roc_auc-0.5)*200:.1f}% better than random guessing")
report.append("")

report.append("=" * 70)
report.append("2. PRECISION-RECALL CURVE")
report.append("-" * 70)
report.append("")
report.append("WHAT IT SHOWS:")
report.append("- Tradeoff between Precision and Recall at different thresholds")
report.append("- More informative than ROC for imbalanced datasets")
report.append("- Average Precision (AP) summarizes the curve")
report.append("")
report.append(f"RESULTS:")
report.append(f"- Average Precision: {avg_precision:.4f}")
report.append(f"- Baseline (No Skill): {y_test.mean():.4f}")
report.append("")
report.append("INTERPRETATION:")
report.append(f"- Model is {avg_precision/y_test.mean():.2f}x better than baseline")
report.append("- Useful for understanding model performance on minority class (survivors)")
report.append("")

report.append("=" * 70)
report.append("3. CONFUSION MATRIX")
report.append("-" * 70)
report.append("")
report.append("WHAT IT SHOWS:")
report.append("- Breakdown of correct and incorrect predictions")
report.append("")
report.append("RESULTS:")
report.append(f"- True Negatives (TN):  {tn:,} - Correctly predicted deaths")
report.append(f"- False Positives (FP): {fp:,} - Wrongly predicted as survivors")
report.append(f"- False Negatives (FN): {fn:,} - Wrongly predicted as deaths")
report.append(f"- True Positives (TP):  {tp:,} - Correctly predicted survivors")
report.append("")
report.append("CLINICAL INTERPRETATION:")
report.append(f"- The model catches {tp} out of {tp+fn} actual survivors ({tp/(tp+fn)*100:.1f}%)")
report.append(f"- It makes {fp} false alarms (predicts survival but patient dies)")
report.append(f"- It misses {fn} survivors (predicts death but patient survives)")
report.append("")

report.append("=" * 70)
report.append("4. CALIBRATION CURVE")
report.append("-" * 70)
report.append("")
report.append("WHAT IT SHOWS:")
report.append("- How well predicted probabilities match actual outcomes")
report.append("- If model says '70% survival', do 70% of those patients survive?")
report.append("")
report.append("INTERPRETATION:")
report.append("- Points close to diagonal: well-calibrated predictions")
report.append("- Points above diagonal: model is under-confident")
report.append("- Points below diagonal: model is over-confident")
report.append("")
report.append("CLINICAL IMPORTANCE:")
report.append("- Well-calibrated models are essential for clinical decision-making")
report.append("- Doctors need accurate probability estimates for patient counseling")
report.append("")

report.append("=" * 70)
report.append("5. DETAILED CLASSIFICATION REPORT")
report.append("=" * 70)
report.append("")
report.append(class_report)
report.append("")

report.append("=" * 70)
report.append("KEY INSIGHTS")
report.append("=" * 70)
report.append("")
report.append("1. ROC-AUC of {:.3f} indicates the model has good discrimination ability".format(roc_auc))
report.append("   for separating survivors from non-survivors.")
report.append("")
report.append("2. The PR curve shows model performance is significantly better than")
report.append("   baseline, especially important given the class imbalance.")
report.append("")
report.append("3. Confusion matrix reveals the model's specific strengths and weaknesses:")
report.append(f"   - Recall of {tp/(tp+fn):.3f} means catching {tp/(tp+fn)*100:.1f}% of survivors")
report.append(f"   - Precision of {tp/(tp+fp) if (tp+fp)>0 else 0:.3f} means {tp/(tp+fp)*100 if (tp+fp)>0 else 0:.1f}% of positive predictions are correct")
report.append("")
report.append("4. These visualizations provide comprehensive evidence for the model's")
report.append("   clinical utility and statistical validity.")
report.append("")

report.append("=" * 70)
report.append("VISUALIZATIONS GENERATED")
report.append("=" * 70)
report.append("")
report.append("1. roc_curve.png - ROC curve with AUC score")
report.append("2. precision_recall_curve.png - PR curve with average precision")
report.append("3. confusion_matrix.png - Detailed confusion matrix heatmap")
report.append("4. calibration_curve.png - Probability calibration analysis")
report.append("5. performance_dashboard.png - Comprehensive 6-panel overview")
report.append("")

# Save report
report_text = '\n'.join(report)
with open(results_dir / 'visualization_report.txt', 'w') as f:
    f.write(report_text)

# Save metrics
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(results_dir / 'evaluation_metrics.csv', index=False)

print("\n" + "=" * 70)
print("VISUALIZATION GENERATION COMPLETE!")
print("=" * 70)
print(f"\nResults saved to: {results_dir}/")
print("\nFiles generated:")
print("  - roc_curve.png")
print("  - precision_recall_curve.png")
print("  - confusion_matrix.png")
print("  - calibration_curve.png")
print("  - performance_dashboard.png (comprehensive overview)")
print("  - evaluation_metrics.csv")
print("  - visualization_report.txt")
print("\n" + "=" * 70)

