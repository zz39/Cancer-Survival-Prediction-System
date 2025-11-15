"""
Regression Models Training - Survival Time Prediction

Trains regression models to predict survival time (in days) for cancer patients.

Models:
- Random Forest Regressor
- XGBoost Regressor
- Linear Regression

Evaluation Metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

print("="*70)
print("REGRESSION MODELS - SURVIVAL TIME PREDICTION")
print("="*70)

# STEP 1: LOAD DATA
print("\n[1/4] Loading data...")

with open('data/lung_cancer_train.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('data/lung_cancer_test.pkl', 'rb') as f:
    test_data = pickle.load(f)

X_train = train_data['X_train']
y_reg_train = train_data['y_reg_train']  # Regression target (survival days)

X_test = test_data['X_test']
y_reg_test = test_data['y_reg_test']

print(f" Training samples: {len(X_train):,}")
print(f" Test samples: {len(X_test):,}")
print(f" Features: {X_train.shape[1]}")
print(f"\nTarget (survival days):")
print(f"  Mean: {y_reg_train.mean():.1f} days")
print(f"  Median: {y_reg_train.median():.1f} days")
print(f"  Range: {y_reg_train.min():.0f} - {y_reg_train.max():.0f} days")

# STEP 2: EVALUATION FUNCTION
def evaluate_regressor(model, X_test, y_test, model_name):
    """Evaluate regression model and return metrics"""
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name}")
    print("-" * 50)
    print(f"MAE:   {mae:.1f} days ({mae/30:.1f} months)")
    print(f"RMSE:  {rmse:.1f} days ({rmse/30:.1f} months)")
    print(f"R² Score: {r2:.3f}")
    
    return {
        'model_name': model_name,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mae_months': mae/30,
        'rmse_months': rmse/30
    }

# STEP 3: TRAIN MODELS
print("\n[2/4] Training regression models...")
print("="*70)

results = []

# Model 1: Linear Regression (Baseline)
print("\n1. Training Linear Regression (Baseline)...")
lr = LinearRegression()
lr.fit(X_train, y_reg_train)
lr_results = evaluate_regressor(lr, X_test, y_reg_test, "Linear Regression")
results.append(lr_results)

# Model 2: Random Forest Regressor
print("\n2. Training Random Forest Regressor...")
rf_reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf_reg.fit(X_train, y_reg_train)
rf_results = evaluate_regressor(rf_reg, X_test, y_reg_test, "Random Forest Regressor")
results.append(rf_results)

# Model 3: XGBoost Regressor
print("\n3. Training XGBoost Regressor...")
xgb_reg = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='rmse'
)
xgb_reg.fit(X_train, y_reg_train)
xgb_results = evaluate_regressor(xgb_reg, X_test, y_reg_test, "XGBoost Regressor")
results.append(xgb_results)

# STEP 4: COMPARE MODELS
print("\n[3/4] Comparing models...")
print("="*70)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('mae')  # Best model has lowest MAE

print("\nRESULTS SUMMARY (sorted by MAE)")
print("-"*70)
print(f"{'Model':<25} {'MAE (days)':<15} {'RMSE (days)':<15} {'R² Score':<10}")
print("-"*70)

for _, row in df_results.iterrows():
    print(f"{row['model_name']:<25} {row['mae']:<15.1f} {row['rmse']:<15.1f} {row['r2']:<10.3f}")

# Find best model
best_model_row = df_results.iloc[0]
best_model_name = best_model_row['model_name']

print("\n" + "="*70)
print(" BEST MODEL FOR SURVIVAL TIME PREDICTION")
print("="*70)
print(f"Model: {best_model_name}")
print(f"MAE:   {best_model_row['mae']:.1f} days (~{best_model_row['mae_months']:.1f} months)")
print(f"RMSE:  {best_model_row['rmse']:.1f} days (~{best_model_row['rmse_months']:.1f} months)")
print(f"R²:    {best_model_row['r2']:.3f}")

# Interpretation
print(f"\n INTERPRETATION:")
print(f"The model predicts survival time with an average error of {best_model_row['mae']:.1f} days")
print(f"(approximately {best_model_row['mae_months']:.1f} months).")
print(f"\nThis means:")
print(f"  • If a patient is predicted to survive 400 days, the actual survival")
print(f"    time is likely between {400-best_model_row['mae']:.0f} and {400+best_model_row['mae']:.0f} days")
print(f"  • The model explains {best_model_row['r2']*100:.1f}% of variance in survival time")

# STEP 5: SAVE BEST MODEL
print("\n[4/4] Saving best model...")

# Select best model object
if best_model_name == "Linear Regression":
    best_model = lr
elif best_model_name == "Random Forest Regressor":
    best_model = rf_reg
else:  # XGBoost
    best_model = xgb_reg

# Save model info
model_info = {
    'model': best_model,
    'model_name': best_model_name,
    'mae': best_model_row['mae'],
    'rmse': best_model_row['rmse'],
    'r2': best_model_row['r2'],
    'mae_months': best_model_row['mae_months']
}

with open('models/best_regressor.pkl', 'wb') as f:
    pickle.dump(model_info, f)
print(" Saved: models/best_regressor.pkl")

# Save results
df_results.to_csv('results/regression_results.csv', index=False)
print(" Saved: results/regression_results.csv")

# Save detailed report
with open('results/regression_report.txt', 'w') as f:
    f.write("SURVIVAL TIME PREDICTION - REGRESSION MODELS\n")
    f.write("="*70 + "\n\n")
    f.write("OBJECTIVE:\n")
    f.write("Predict how many days a cancer patient will survive after diagnosis.\n\n")
    f.write("MODELS TESTED:\n")
    f.write("1. Linear Regression (Baseline)\n")
    f.write("2. Random Forest Regressor\n")
    f.write("3. XGBoost Regressor\n\n")
    f.write("RESULTS:\n")
    f.write("-"*70 + "\n")
    f.write(df_results.to_string(index=False) + "\n\n")
    f.write("="*70 + "\n")
    f.write("BEST MODEL\n")
    f.write("="*70 + "\n")
    f.write(f"Model: {best_model_name}\n")
    f.write(f"MAE:   {best_model_row['mae']:.1f} days (~{best_model_row['mae_months']:.1f} months)\n")
    f.write(f"RMSE:  {best_model_row['rmse']:.1f} days (~{best_model_row['rmse_months']:.1f} months)\n")
    f.write(f"R²:    {best_model_row['r2']:.3f}\n\n")
    f.write("INTERPRETATION:\n")
    f.write(f"The {best_model_name} can predict survival time with an average\n")
    f.write(f"error of {best_model_row['mae']:.1f} days (about {best_model_row['mae_months']:.1f} months).\n\n")
    f.write("CLINICAL USE:\n")
    f.write("- Treatment planning (how long to plan care for)\n")
    f.write("- Resource allocation (hospital bed planning)\n")
    f.write("- Patient counseling (realistic expectations)\n")
    f.write("- Clinical trial enrollment decisions\n")

print(" Saved: results/regression_report.txt")

# Feature importance for best tree-based model
if best_model_name in ["Random Forest Regressor", "XGBoost Regressor"]:
    print("\nTop 10 Most Important Features:")
    print("-" * 50)
    
    with open('data/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    if best_model_name == "Random Forest Regressor":
        importances = best_model.feature_importances_
    else:  # XGBoost
        importances = best_model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:<25} {row['importance']:.4f}")
    
    # Save feature importance
    feature_importance.to_csv('results/regression_feature_importance.csv', index=False)
    print("\n Saved: results/regression_feature_importance.csv")

print("\n" + "="*70)
print(" REGRESSION TRAINING COMPLETE!")
print("="*70)
print(f"\nSUMMARY:")
print(f" Trained 3 regression models")
print(f" Best model: {best_model_name}")
print(f" Prediction error: ±{best_model_row['mae']:.1f} days (~{best_model_row['mae_months']:.1f} months)")
print(f" Model saved and ready for use")
print("="*70)

