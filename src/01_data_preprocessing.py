"""
Data Preprocessing Script for Lung Cancer Survival Prediction

This script loads, cleans, and prepares the data for model training.

Outputs:
- lung_cancer_train.pkl: Training data (features + labels)
- lung_cancer_test.pkl: Test data (features + labels)
- pollution_processed.pkl: Processed pollution dataset
- feature_names.pkl: Feature column names
- encoders.pkl: Label encoders for decoding predictions
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("LUNG CANCER SURVIVAL PREDICTION - DATA PREPROCESSING")
print("="*70)

# STEP 1: LOAD DATASETS
print("\n[1/7] Loading datasets...")

# Use sample size for faster processing during development
# Set SAMPLE_SIZE = None to use full dataset
SAMPLE_SIZE = 100000  # Use 100K samples for development

df_lung = pd.read_csv('data/Lung Cancer.csv', nrows=SAMPLE_SIZE)
df_pollution = pd.read_csv('data/cancer patient data sets.csv')

if SAMPLE_SIZE:
    print(f" Lung Cancer Dataset: {df_lung.shape[0]:,} patients (sampled from 890K), {df_lung.shape[1]} features")
else:
    print(f" Lung Cancer Dataset: {df_lung.shape[0]:,} patients, {df_lung.shape[1]} features")
print(f" Pollution Dataset: {df_pollution.shape[0]:,} patients, {df_pollution.shape[1]} features")

# STEP 2: CLEAN LUNG CANCER DATASET
print("\n[2/7] Cleaning Lung Cancer data...")

# Convert dates
df_lung['diagnosis_date'] = pd.to_datetime(df_lung['diagnosis_date'])
df_lung['end_treatment_date'] = pd.to_datetime(df_lung['end_treatment_date'])

# Calculate survival days
df_lung['survival_days'] = (df_lung['end_treatment_date'] - df_lung['diagnosis_date']).dt.days

# Remove invalid records
initial_count = len(df_lung)
df_lung_clean = df_lung[(df_lung['survival_days'] > 0) & (df_lung['survival_days'] < 2000)].copy()
removed_count = initial_count - len(df_lung_clean)

print(f" Removed {removed_count:,} rows with invalid survival_days")
print(f" Final dataset: {len(df_lung_clean):,} patients")
print(f" Survival rate: {df_lung_clean['survived'].mean():.2%}")
print(f" Death rate: {1 - df_lung_clean['survived'].mean():.2%}")

# Check for missing values
missing_values = df_lung_clean.isnull().sum().sum()
print(f" Missing values: {missing_values}")

# STEP 3: ENCODE CATEGORICAL FEATURES
print("\n[3/7] Encoding categorical features...")

# Create encoders dictionary to save later
encoders = {}

# Encode cancer_stage
le_stage = LabelEncoder()
df_lung_clean['cancer_stage_encoded'] = le_stage.fit_transform(df_lung_clean['cancer_stage'])
encoders['cancer_stage'] = le_stage
print(f" cancer_stage: {list(le_stage.classes_)}")

# Encode treatment_type
le_treatment = LabelEncoder()
df_lung_clean['treatment_type_encoded'] = le_treatment.fit_transform(df_lung_clean['treatment_type'])
encoders['treatment_type'] = le_treatment
print(f" treatment_type: {list(le_treatment.classes_)}")

# Encode smoking_status
le_smoking = LabelEncoder()
df_lung_clean['smoking_status_encoded'] = le_smoking.fit_transform(df_lung_clean['smoking_status'])
encoders['smoking_status'] = le_smoking
print(f" smoking_status: {list(le_smoking.classes_)}")

# Encode family_history
le_family = LabelEncoder()
df_lung_clean['family_history_encoded'] = le_family.fit_transform(df_lung_clean['family_history'])
encoders['family_history'] = le_family
print(f" family_history: {list(le_family.classes_)}")

# Encode gender
le_gender = LabelEncoder()
df_lung_clean['gender_encoded'] = le_gender.fit_transform(df_lung_clean['gender'])
encoders['gender'] = le_gender
print(f" gender: {list(le_gender.classes_)}")

# STEP 4: CREATE FEATURE SETS
print("\n[4/7] Creating feature sets...")

# Select features for modeling
feature_columns = [
    'age',
    'bmi',
    'cancer_stage_encoded',
    'treatment_type_encoded',
    'smoking_status_encoded',
    'cholesterol_level',
    'hypertension',
    'asthma',
    'cirrhosis',
    'other_cancer',
    'family_history_encoded',
    'gender_encoded'
]

# Prepare features (X) and targets (y)
X = df_lung_clean[feature_columns].copy()
y_classification = df_lung_clean['survived'].copy()  # Binary: 0 or 1
y_regression = df_lung_clean['survival_days'].copy()  # Continuous: days

print(f" Features shape: {X.shape}")
print(f" Classification target shape: {y_classification.shape}")
print(f" Regression target shape: {y_regression.shape}")

# Save feature names for later use
feature_names = feature_columns
print(f" Feature names saved: {len(feature_names)} features")

# STEP 5: SPLIT INTO TRAIN/TEST SETS
print("\n[5/7] Creating train/test splits...")

# Use 20% of data for testing (we have a lot of data, so 20% is sufficient)
# Use stratify to maintain class balance in train/test
X_train, X_test, y_class_train, y_class_test = train_test_split(
    X, y_classification, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_classification
)

# Split regression target the same way
_, _, y_reg_train, y_reg_test = train_test_split(
    X, y_regression,
    test_size=0.2,
    random_state=42,
    stratify=y_classification  # Use same stratification for consistency
)

print(f" Training set: {len(X_train):,} samples")
print(f" Test set: {len(X_test):,} samples")
print(f" Train survival rate: {y_class_train.mean():.2%}")
print(f" Test survival rate: {y_class_test.mean():.2%}")

# STEP 6: PROCESS AIR POLLUTION DATASET
print("\n[6/7] Processing Air Pollution dataset...")

# Create interaction features
df_pollution['air_pollution_x_smoking'] = df_pollution['Air Pollution'] * df_pollution['Smoking']
df_pollution['air_pollution_x_occupation'] = df_pollution['Air Pollution'] * df_pollution['OccuPational Hazards']

print(f" Created interaction features")

# Encode risk levels
le_risk = LabelEncoder()
df_pollution['risk_level_encoded'] = le_risk.fit_transform(df_pollution['Level'])
encoders['risk_level'] = le_risk
print(f" Risk levels: {list(le_risk.classes_)}")

# Prepare pollution features
pollution_features = ['Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy',
                      'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
                      'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker',
                      'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss',
                      'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
                      'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough',
                      'Snoring', 'air_pollution_x_smoking', 'air_pollution_x_occupation']

X_pollution = df_pollution[pollution_features].copy()
y_pollution = df_pollution['risk_level_encoded'].copy()

print(f" Pollution features: {X_pollution.shape}")
print(f" Risk distribution: Low={sum(y_pollution==0)}, Medium={sum(y_pollution==1)}, High={sum(y_pollution==2)}")

# STEP 7: SAVE PROCESSED DATA
print("\n[7/7] Saving processed data...")

# Save lung cancer train/test data
train_data = {
    'X_train': X_train,
    'y_class_train': y_class_train,
    'y_reg_train': y_reg_train
}

test_data = {
    'X_test': X_test,
    'y_class_test': y_class_test,
    'y_reg_test': y_reg_test
}

pollution_data = {
    'X_pollution': X_pollution,
    'y_pollution': y_pollution
}

with open('data/lung_cancer_train.pkl', 'wb') as f:
    pickle.dump(train_data, f)
print(" Saved: lung_cancer_train.pkl")

with open('data/lung_cancer_test.pkl', 'wb') as f:
    pickle.dump(test_data, f)
print(" Saved: lung_cancer_test.pkl")

with open('data/pollution_processed.pkl', 'wb') as f:
    pickle.dump(pollution_data, f)
print(" Saved: pollution_processed.pkl")

with open('data/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print(" Saved: feature_names.pkl")

with open('data/encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
print(" Saved: encoders.pkl")

# Also save the clean dataframe for treatment analysis
with open('data/lung_cancer_clean.pkl', 'wb') as f:
    pickle.dump(df_lung_clean, f)
print(" Saved: lung_cancer_clean.pkl (for treatment analysis)")

# SUMMARY
print("\n" + "="*70)
print("DATA PREPROCESSING COMPLETE!")
print("="*70)
print("\nSummary:")
print(f"  Total patients processed: {len(df_lung_clean):,}")
print(f"  Training samples: {len(X_train):,}")
print(f"  Test samples: {len(X_test):,}")
print(f"  Features: {len(feature_names)}")
print(f"  Class balance: {y_class_train.mean():.1%} survival, {1-y_class_train.mean():.1%} death")
print(f"\nFiles saved in 'data/' directory:")
print(f"  - lung_cancer_train.pkl")
print(f"  - lung_cancer_test.pkl")
print(f"  - pollution_processed.pkl")
print(f"  - feature_names.pkl")
print(f"  - encoders.pkl")
print(f"  - lung_cancer_clean.pkl")
print("\n Ready for model training!")
print("="*70)

