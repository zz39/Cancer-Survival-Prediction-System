"""
Class Imbalance Handling Script

This script applies different techniques to handle the severe class imbalance
(78% deaths vs 22% survivors) in the training data.

Techniques implemented:
1. Class Weights - Give more importance to minority class
2. SMOTE - Synthetic Minority Over-sampling
3. Tomek Links - Remove borderline samples
4. SMOTETomek - Combined approach (SMOTE + Tomek)

Outputs:
- train_with_smote.pkl: SMOTE balanced training data
- train_with_tomek.pkl: Tomek Links cleaned training data
- train_with_combined.pkl: SMOTETomek balanced training data
- class_weights.pkl: Calculated class weights for models
- imbalance_stats.pkl: Statistics before/after each technique
"""

import pandas as pd
import numpy as np
import pickle
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CLASS IMBALANCE HANDLING")
print("="*70)

# STEP 1: LOAD TRAINING DATA
print("\n[1/6] Loading training data...")

with open('data/lung_cancer_train.pkl', 'rb') as f:
    train_data = pickle.load(f)

X_train = train_data['X_train']
y_train = train_data['y_class_train']

print(f" Training samples: {len(X_train):,}")
print(f" Features: {X_train.shape[1]}")

# STEP 2: ANALYZE ORIGINAL IMBALANCE
print("\n[2/6] Analyzing original class distribution...")

original_counts = Counter(y_train)
total = len(y_train)

deaths = original_counts[0]
survivors = original_counts[1]
imbalance_ratio = deaths / survivors

print(f"\nOriginal Distribution:")
print(f"  Deaths (0):    {deaths:,} samples ({deaths/total*100:.1f}%)")
print(f"  Survivors (1): {survivors:,} samples ({survivors/total*100:.1f}%)")
print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")
print(f"\n  Problem: Models will predict death {deaths/total*100:.1f}% of the time!")

# STEP 3: CALCULATE CLASS WEIGHTS
print("\n[3/6] Calculating class weights...")

# Calculate balanced class weights
# Formula: total / (n_classes * count_for_class)
n_classes = 2
weight_for_deaths = total / (n_classes * deaths)
weight_for_survivors = total / (n_classes * survivors)

class_weights = {
    0: weight_for_deaths,
    1: weight_for_survivors
}

print(f"\nClass Weights (for model parameters):")
print(f"  Deaths (0):    {weight_for_deaths:.3f}")
print(f"  Survivors (1): {weight_for_survivors:.3f}")
print(f"   Survivors are {weight_for_survivors/weight_for_deaths:.2f}x more important")

# Also calculate for scikit-learn format
class_weight_dict = {0: weight_for_deaths, 1: weight_for_survivors}

# STEP 4: APPLY SMOTE (Synthetic Minority Over-sampling)
print("\n[4/6] Applying SMOTE...")
print("  Creating synthetic survivor samples...")

# Use SMOTE with sampling_strategy=0.8 (make survivors 80% of deaths)
# This gives roughly 44% survivors, 56% deaths - more balanced
smote = SMOTE(sampling_strategy=0.8, random_state=42, k_neighbors=5)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

smote_counts = Counter(y_train_smote)
smote_total = len(y_train_smote)

print(f"\n SMOTE Complete!")
print(f"  Deaths (0):    {smote_counts[0]:,} samples ({smote_counts[0]/smote_total*100:.1f}%)")
print(f"  Survivors (1): {smote_counts[1]:,} samples ({smote_counts[1]/smote_total*100:.1f}%)")
print(f"  New samples created: {len(X_train_smote) - len(X_train):,}")
print(f"  New Imbalance Ratio: {smote_counts[0]/smote_counts[1]:.2f}:1")

# STEP 5: APPLY TOMEK LINKS (Boundary Cleaning)
print("\n[5/6] Applying Tomek Links...")
print("  Removing confusing boundary samples...")

tomek = TomekLinks()
X_train_tomek, y_train_tomek = tomek.fit_resample(X_train, y_train)

tomek_counts = Counter(y_train_tomek)
tomek_total = len(y_train_tomek)

print(f"\n Tomek Links Complete!")
print(f"  Deaths (0):    {tomek_counts[0]:,} samples ({tomek_counts[0]/tomek_total*100:.1f}%)")
print(f"  Survivors (1): {tomek_counts[1]:,} samples ({tomek_counts[1]/tomek_total*100:.1f}%)")
print(f"  Samples removed: {len(X_train) - len(X_train_tomek):,}")
print(f"   Cleaner decision boundary!")

# STEP 6: APPLY SMOTETOMEK (Combined Approach)
print("\n[6/6] Applying SMOTETomek (Combined)...")
print("  Step 1: Creating synthetic samples...")
print("  Step 2: Cleaning boundaries...")

smt = SMOTETomek(sampling_strategy=0.8, random_state=42)
X_train_combined, y_train_combined = smt.fit_resample(X_train, y_train)

combined_counts = Counter(y_train_combined)
combined_total = len(y_train_combined)

print(f"\n SMOTETomek Complete!")
print(f"  Deaths (0):    {combined_counts[0]:,} samples ({combined_counts[0]/combined_total*100:.1f}%)")
print(f"  Survivors (1): {combined_counts[1]:,} samples ({combined_counts[1]/combined_total*100:.1f}%)")
print(f"  Net change: {len(X_train_combined) - len(X_train):,} samples")
print(f"  New Imbalance Ratio: {combined_counts[0]/combined_counts[1]:.2f}:1")

# STEP 7: SAVE ALL VERSIONS
print("\n[7/7] Saving balanced datasets...")

# Save SMOTE version
smote_data = {
    'X_train': X_train_smote,
    'y_train': y_train_smote,
    'technique': 'SMOTE',
    'original_size': len(X_train),
    'new_size': len(X_train_smote)
}
with open('data/train_with_smote.pkl', 'wb') as f:
    pickle.dump(smote_data, f)
print(" Saved: train_with_smote.pkl")

# Save Tomek version
tomek_data = {
    'X_train': X_train_tomek,
    'y_train': y_train_tomek,
    'technique': 'Tomek Links',
    'original_size': len(X_train),
    'new_size': len(X_train_tomek)
}
with open('data/train_with_tomek.pkl', 'wb') as f:
    pickle.dump(tomek_data, f)
print(" Saved: train_with_tomek.pkl")

# Save SMOTETomek version
combined_data = {
    'X_train': X_train_combined,
    'y_train': y_train_combined,
    'technique': 'SMOTETomek',
    'original_size': len(X_train),
    'new_size': len(X_train_combined)
}
with open('data/train_with_combined.pkl', 'wb') as f:
    pickle.dump(combined_data, f)
print(" Saved: train_with_combined.pkl")

# Save class weights
weights_data = {
    'class_weight_dict': class_weight_dict,
    'weight_for_deaths': weight_for_deaths,
    'weight_for_survivors': weight_for_survivors,
    'sklearn_format': 'balanced'  # Can use 'balanced' parameter in sklearn
}
with open('data/class_weights.pkl', 'wb') as f:
    pickle.dump(weights_data, f)
print(" Saved: class_weights.pkl")

# Save statistics for reporting
stats = {
    'original': {
        'deaths': deaths,
        'survivors': survivors,
        'total': total,
        'ratio': imbalance_ratio
    },
    'smote': {
        'deaths': smote_counts[0],
        'survivors': smote_counts[1],
        'total': smote_total,
        'ratio': smote_counts[0]/smote_counts[1]
    },
    'tomek': {
        'deaths': tomek_counts[0],
        'survivors': tomek_counts[1],
        'total': tomek_total,
        'ratio': tomek_counts[0]/tomek_counts[1]
    },
    'combined': {
        'deaths': combined_counts[0],
        'survivors': combined_counts[1],
        'total': combined_total,
        'ratio': combined_counts[0]/combined_counts[1]
    }
}
with open('data/imbalance_stats.pkl', 'wb') as f:
    pickle.dump(stats, f)
print(" Saved: imbalance_stats.pkl")

# SUMMARY
print("\n" + "="*70)
print("CLASS IMBALANCE HANDLING COMPLETE!")
print("="*70)

print("\n COMPARISON SUMMARY:")
print("-" * 70)
print(f"{'Technique':<20} {'Deaths':<15} {'Survivors':<15} {'Ratio':<10} {'Total':<10}")
print("-" * 70)
print(f"{'Original':<20} {deaths:<15,} {survivors:<15,} {imbalance_ratio:<10.2f} {total:<10,}")
print(f"{'SMOTE':<20} {smote_counts[0]:<15,} {smote_counts[1]:<15,} {smote_counts[0]/smote_counts[1]:<10.2f} {smote_total:<10,}")
print(f"{'Tomek Links':<20} {tomek_counts[0]:<15,} {tomek_counts[1]:<15,} {tomek_counts[0]/tomek_counts[1]:<10.2f} {tomek_total:<10,}")
print(f"{'SMOTETomek':<20} {combined_counts[0]:<15,} {combined_counts[1]:<15,} {combined_counts[0]/combined_counts[1]:<10.2f} {combined_total:<10,}")
print("-" * 70)

print("\n RECOMMENDATIONS:")
print("  1. Use SMOTE or SMOTETomek for Logistic Regression")
print("  2. Use class_weight='balanced' for Random Forest/XGBoost")
print("  3. Try all versions and compare results")
print("  4. SMOTETomek usually gives best overall performance")

print("\n Ready for model training with balanced data!")
print("="*70)

