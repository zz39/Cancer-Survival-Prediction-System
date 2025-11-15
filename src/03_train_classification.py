"""
Classification Model Training Script

Trains multiple models with different class imbalance handling techniques
for lung cancer survival prediction.

Models trained:
1. Logistic Regression (baseline, SMOTE, SMOTETomek)
2. Random Forest (class weights, SMOTE, SMOTETomek)
3. XGBoost (scale_pos_weight, SMOTE, SMOTETomek)

Outputs:
- Trained models saved in models/
- Performance comparison table in results/
- Visualizations in results/plots/
"""

import pandas as pd
import numpy as np
import pickle
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost (may fail if OpenMP not installed)
XGBOOST_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as e:
    print(f"  XGBoost not available: {str(e)[:100]}")
    print("   Will train Logistic Regression and Random Forest only.")

print("="*70)
print("CLASSIFICATION MODEL TRAINING - SURVIVAL PREDICTION")
print("="*70)

# STEP 1: LOAD DATA
print("\n[1/5] Loading training and test data...")

# Load original training data
with open('data/lung_cancer_train.pkl', 'rb') as f:
    train_original = pickle.load(f)
X_train_original = train_original['X_train']
y_train_original = train_original['y_class_train']

# Load test data
with open('data/lung_cancer_test.pkl', 'rb') as f:
    test_data = pickle.load(f)
X_test = test_data['X_test']
y_test = test_data['y_class_test']

# Load SMOTE balanced data
with open('data/train_with_smote.pkl', 'rb') as f:
    train_smote = pickle.load(f)
X_train_smote = train_smote['X_train']
y_train_smote = train_smote['y_train']

# Load SMOTETomek balanced data
with open('data/train_with_combined.pkl', 'rb') as f:
    train_combined = pickle.load(f)
X_train_combined = train_combined['X_train']
y_train_combined = train_combined['y_train']

# Load class weights
with open('data/class_weights.pkl', 'rb') as f:
    weights_data = pickle.load(f)
class_weight_dict = weights_data['class_weight_dict']

print(f" Original training: {len(X_train_original):,} samples")
print(f" SMOTE training: {len(X_train_smote):,} samples")
print(f" SMOTETomek training: {len(X_train_combined):,} samples")
print(f" Test set: {len(X_test):,} samples")

# STEP 2: DEFINE EVALUATION FUNCTION
print("\n[2/5] Setting up evaluation metrics...")

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics"""
    start_time = time.time()
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    train_time = time.time() - start_time
    
    # Print results
    print(f"\n  {model_name}")
    print(f"  {'='*60}")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f} ⭐")
    print(f"  F1-Score:  {f1:.3f}")
    print(f"  ROC-AUC:   {roc_auc:.3f}")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'training_time': train_time
    }

print(" Evaluation function ready")

# STEP 3: TRAIN LOGISTIC REGRESSION MODELS
print("\n[3/5] Training Logistic Regression models...")
print("-" * 70)

results = []

# 3.1: Logistic Regression - Baseline
print("\n[3.1] Logistic Regression - Baseline (no balancing)")
lr_baseline = LogisticRegression(max_iter=1000, random_state=42)
lr_baseline.fit(X_train_original, y_train_original)
result = evaluate_model(lr_baseline, X_test, y_test, "Logistic Regression - Baseline")
results.append(result)

# 3.2: Logistic Regression - SMOTE
print("\n[3.2] Logistic Regression - SMOTE")
lr_smote = LogisticRegression(max_iter=1000, random_state=42)
lr_smote.fit(X_train_smote, y_train_smote)
result = evaluate_model(lr_smote, X_test, y_test, "Logistic Regression - SMOTE")
results.append(result)

# 3.3: Logistic Regression - SMOTETomek
print("\n[3.3] Logistic Regression - SMOTETomek")
lr_combined = LogisticRegression(max_iter=1000, random_state=42)
lr_combined.fit(X_train_combined, y_train_combined)
result = evaluate_model(lr_combined, X_test, y_test, "Logistic Regression - SMOTETomek")
results.append(result)

# STEP 4: TRAIN RANDOM FOREST MODELS
print("\n[4/5] Training Random Forest models...")
print("-" * 70)

# 4.1: Random Forest - Class Weights
print("\n[4.1] Random Forest - Class Weights")
rf_weighted = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_weighted.fit(X_train_original, y_train_original)
result = evaluate_model(rf_weighted, X_test, y_test, "Random Forest - Class Weights")
results.append(result)

# 4.2: Random Forest - SMOTE
print("\n[4.2] Random Forest - SMOTE")
rf_smote = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_smote.fit(X_train_smote, y_train_smote)
result = evaluate_model(rf_smote, X_test, y_test, "Random Forest - SMOTE")
results.append(result)

# 4.3: Random Forest - SMOTETomek
print("\n[4.3] Random Forest - SMOTETomek")
rf_combined = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_combined.fit(X_train_combined, y_train_combined)
result = evaluate_model(rf_combined, X_test, y_test, "Random Forest - SMOTETomek")
results.append(result)

# STEP 5: TRAIN XGBOOST MODELS (if available)
if XGBOOST_AVAILABLE:
    print("\n[5/5] Training XGBoost models...")
    print("-" * 70)

    # Calculate scale_pos_weight for XGBoost
    scale_pos_weight = len(y_train_original[y_train_original==0]) / len(y_train_original[y_train_original==1])

    # 5.1: XGBoost - scale_pos_weight
    print(f"\n[5.1] XGBoost - scale_pos_weight ({scale_pos_weight:.2f})")
    xgb_weighted = XGBClassifier(
        n_estimators=100,
        scale_pos_weight=scale_pos_weight,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_weighted.fit(X_train_original, y_train_original)
    result = evaluate_model(xgb_weighted, X_test, y_test, "XGBoost - scale_pos_weight")
    results.append(result)

    # 5.2: XGBoost - SMOTE
    print("\n[5.2] XGBoost - SMOTE")
    xgb_smote = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_smote.fit(X_train_smote, y_train_smote)
    result = evaluate_model(xgb_smote, X_test, y_test, "XGBoost - SMOTE")
    results.append(result)

    # 5.3: XGBoost - SMOTETomek
    print("\n[5.3] XGBoost - SMOTETomek")
    xgb_combined = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_combined.fit(X_train_combined, y_train_combined)
    result = evaluate_model(xgb_combined, X_test, y_test, "XGBoost - SMOTETomek")
    results.append(result)
else:
    print("\n[5/5] Skipping XGBoost models (library not available)")
    xgb_weighted = xgb_smote = xgb_combined = None

# STEP 6: COMPARE RESULTS
print("\n" + "="*70)
print("MODEL COMPARISON - SURVIVAL PREDICTION")
print("="*70)

# Create comparison DataFrame
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('recall', ascending=False)

print("\n Full Results Table (sorted by Recall):")
print("-" * 70)
print(df_results[['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].to_string(index=False))

# Find best models
best_recall = df_results.iloc[0]
best_f1 = df_results.sort_values('f1_score', ascending=False).iloc[0]
best_roc = df_results.sort_values('roc_auc', ascending=False).iloc[0]

print("\n BEST MODELS:")
print(f"  Best Recall:    {best_recall['model_name']} ({best_recall['recall']:.3f})")
print(f"  Best F1-Score:  {best_f1['model_name']} ({best_f1['f1_score']:.3f})")
print(f"  Best ROC-AUC:   {best_roc['model_name']} ({best_roc['roc_auc']:.3f})")

# STEP 7: SAVE RESULTS AND MODELS
print("\n[7/7] Saving results and best models...")

# Save comparison table
df_results.to_csv('results/classification_results.csv', index=False)
print(" Saved: results/classification_results.csv")

# Save best model by recall (most important for clinical use)
best_model_name = best_recall['model_name']
if 'Logistic' in best_model_name:
    if 'SMOTE' in best_model_name and 'Tomek' in best_model_name:
        best_model = lr_combined
    elif 'SMOTE' in best_model_name:
        best_model = lr_smote
    else:
        best_model = lr_baseline
elif 'Random Forest' in best_model_name:
    if 'SMOTE' in best_model_name and 'Tomek' in best_model_name:
        best_model = rf_combined
    elif 'SMOTE' in best_model_name:
        best_model = rf_smote
    else:
        best_model = rf_weighted
else:  # XGBoost
    if 'SMOTE' in best_model_name and 'Tomek' in best_model_name:
        best_model = xgb_combined
    elif 'SMOTE' in best_model_name:
        best_model = xgb_smote
    else:
        best_model = xgb_weighted

with open('models/best_classifier.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f" Saved best model: {best_model_name}")

# Save all trained models
models_dict = {
    'lr_baseline': lr_baseline,
    'lr_smote': lr_smote,
    'lr_combined': lr_combined,
    'rf_weighted': rf_weighted,
    'rf_smote': rf_smote,
    'rf_combined': rf_combined,
}

if XGBOOST_AVAILABLE:
    models_dict.update({
        'xgb_weighted': xgb_weighted,
        'xgb_smote': xgb_smote,
        'xgb_combined': xgb_combined
    })

with open('models/all_classifiers.pkl', 'wb') as f:
    pickle.dump(models_dict, f)
print(" Saved: models/all_classifiers.pkl")

# SUMMARY
print("\n" + "="*70)
print("CLASSIFICATION TRAINING COMPLETE!")
print("="*70)

num_models = len(results)
print(f"\n Trained {num_models} models:")
print(f"   - 3 Logistic Regression variants")
print(f"   - 3 Random Forest variants")
if XGBOOST_AVAILABLE:
    print(f"   - 3 XGBoost variants")
else:
    print(f"   - XGBoost skipped (install OpenMP: brew install libomp)")

print(f"\n Performance Improvement:")
baseline_recall = df_results[df_results['model_name'].str.contains('Baseline')]['recall'].values[0]
best_recall_val = best_recall['recall']
improvement = (best_recall_val - baseline_recall) / baseline_recall * 100 if baseline_recall > 0 else float('inf')

print(f"   Baseline Recall: {baseline_recall:.1%}")
print(f"   Best Recall:     {best_recall_val:.1%}")
if improvement != float('inf'):
    print(f"   Improvement:     {improvement:.0f}% ")
else:
    print(f"   Improvement:     Infinite (baseline was 0%!) ")

print(f"\n Recommended Model for Production:")
print(f"   {best_model_name}")
print(f"   Recall: {best_recall['recall']:.1%} | F1: {best_recall['f1_score']:.3f} | ROC-AUC: {best_recall['roc_auc']:.3f}")

print("\n Results saved in 'results/' directory")
print(" Models saved in 'models/' directory")
print("="*70)

