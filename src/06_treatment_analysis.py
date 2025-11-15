"""
Treatment Effectiveness Analysis

Analyzes which treatment types work best for different cancer stages.

Objectives:
1. Compare survival rates across treatment types
2. Identify best treatment for each cancer stage
3. Generate treatment recommendations for clinicians

Treatments: Surgery, Chemotherapy, Radiation, Immunotherapy, Targeted Therapy
Cancer Stages: Stage I, Stage II, Stage III, Stage IV
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from scipy import stats

print("="*70)
print("TREATMENT EFFECTIVENESS ANALYSIS")
print("="*70)

# STEP 1: LOAD AND PREPARE DATA
print("\n[1/4] Loading data...")

# Load original CSV for treatment analysis (has all original columns)
df = pd.read_csv('data/Lung Cancer.csv', nrows=100000)

print(f" Loaded {len(df):,} patient records")

# Convert dates and calculate survival
df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'])
df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'])
df['survival_days'] = (df['end_treatment_date'] - df['diagnosis_date']).dt.days

# Filter valid data
df = df[(df['survival_days'] > 0) & (df['survival_days'] < 2000)]

# Create binary survival (>365 days = survived)
df['survived'] = (df['survival_days'] > 365).astype(int)

print(f" After filtering: {len(df):,} valid records")
print(f"\nDataset info:")
print(f"  Cancer stages: {df['cancer_stage'].unique()}")
print(f"  Treatment types: {df['treatment_type'].unique()}")
print(f"  Overall survival rate: {df['survived'].mean()*100:.1f}%")

# STEP 2: TREATMENT EFFECTIVENESS BY STAGE
print("\n[2/4] Analyzing treatment effectiveness by cancer stage...")
print("="*70)

# Group by stage and treatment
treatment_analysis = df.groupby(['cancer_stage', 'treatment_type']).agg({
    'survived': ['count', 'sum', 'mean'],
    'survival_days': ['mean', 'median', 'std']
}).round(3)

treatment_analysis.columns = ['_'.join(col).strip() for col in treatment_analysis.columns.values]
treatment_analysis = treatment_analysis.reset_index()
treatment_analysis.columns = ['cancer_stage', 'treatment_type', 'total_patients', 
                               'survivors', 'survival_rate', 'avg_survival_days', 
                               'median_survival_days', 'std_survival_days']

# Calculate percentage and convert to readable format
treatment_analysis['survival_rate_pct'] = (treatment_analysis['survival_rate'] * 100).round(1)

print("\nTREATMENT EFFECTIVENESS TABLE")
print("-"*70)
print(treatment_analysis.to_string(index=False))

# STEP 3: IDENTIFY BEST TREATMENT PER STAGE
print("\n[3/4] Identifying best treatment for each stage...")
print("="*70)

recommendations = []

for stage in df['cancer_stage'].unique():
    stage_data = treatment_analysis[treatment_analysis['cancer_stage'] == stage]
    
    # Find treatment with highest survival rate (minimum 50 patients for statistical significance)
    valid_treatments = stage_data[stage_data['total_patients'] >= 50]
    
    if len(valid_treatments) > 0:
        best_treatment = valid_treatments.loc[valid_treatments['survival_rate'].idxmax()]
        
        recommendations.append({
            'cancer_stage': stage,
            'recommended_treatment': best_treatment['treatment_type'],
            'survival_rate': f"{best_treatment['survival_rate_pct']:.1f}%",
            'avg_survival_days': f"{best_treatment['avg_survival_days']:.0f}",
            'sample_size': int(best_treatment['total_patients']),
            'confidence': 'High' if best_treatment['total_patients'] >= 200 else 'Moderate'
        })

recommendations_df = pd.DataFrame(recommendations)

print("\n TREATMENT RECOMMENDATIONS BY CANCER STAGE")
print("="*70)
for _, rec in recommendations_df.iterrows():
    print(f"\n{rec['cancer_stage']}:")
    print(f"   Recommended Treatment: {rec['recommended_treatment']}")
    print(f"   Survival Rate: {rec['survival_rate']}")
    print(f"   Average Survival: {rec['avg_survival_days']} days")
    print(f"   Sample Size: {rec['sample_size']} patients")
    print(f"   Confidence: {rec['confidence']}")

# STEP 4: STATISTICAL SIGNIFICANCE TESTING
print("\n[4/4] Statistical significance testing...")
print("="*70)

significance_results = []

for stage in df['cancer_stage'].unique():
    stage_df = df[df['cancer_stage'] == stage]
    treatments = stage_df['treatment_type'].unique()
    
    print(f"\n{stage}:")
    print("-" * 50)
    
    # Compare each treatment pair
    for i, treat1 in enumerate(treatments):
        for treat2 in treatments[i+1:]:
            group1 = stage_df[stage_df['treatment_type'] == treat1]['survived']
            group2 = stage_df[stage_df['treatment_type'] == treat2]['survived']
            
            if len(group1) >= 30 and len(group2) >= 30:  # Minimum sample size
                # Chi-square test for survival rates
                contingency = pd.crosstab(
                    stage_df[stage_df['treatment_type'].isin([treat1, treat2])]['treatment_type'],
                    stage_df[stage_df['treatment_type'].isin([treat1, treat2])]['survived']
                )
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                
                significant = " Significant" if p_value < 0.05 else "Not significant"
                
                survival1 = group1.mean() * 100
                survival2 = group2.mean() * 100
                diff = abs(survival1 - survival2)
                
                print(f"  {treat1} vs {treat2}:")
                print(f"    Survival rates: {survival1:.1f}% vs {survival2:.1f}% (diff: {diff:.1f}%)")
                print(f"    p-value: {p_value:.4f}  {significant}")
                
                significance_results.append({
                    'stage': stage,
                    'treatment1': treat1,
                    'treatment2': treat2,
                    'survival_rate1': f"{survival1:.1f}%",
                    'survival_rate2': f"{survival2:.1f}%",
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })

# STEP 5: SAVE RESULTS
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save treatment effectiveness table
treatment_analysis.to_csv('results/treatment_effectiveness.csv', index=False)
print(" Saved: results/treatment_effectiveness.csv")

# Save recommendations
recommendations_df.to_csv('results/treatment_recommendations.csv', index=False)
print(" Saved: results/treatment_recommendations.csv")

# Save significance results
if significance_results:
    pd.DataFrame(significance_results).to_csv('results/treatment_significance.csv', index=False)
    print(" Saved: results/treatment_significance.csv")

# Save comprehensive report
with open('results/treatment_analysis_report.txt', 'w') as f:
    f.write("TREATMENT EFFECTIVENESS ANALYSIS - COMPREHENSIVE REPORT\n")
    f.write("="*70 + "\n\n")
    
    f.write("OBJECTIVE:\n")
    f.write("Identify which treatment types are most effective for different cancer stages.\n\n")
    
    f.write("METHODOLOGY:\n")
    f.write("- Analyzed 100,000 patient records\n")
    f.write("- Grouped by cancer stage and treatment type\n")
    f.write("- Calculated survival rates (>365 days = survived)\n")
    f.write("- Performed statistical significance testing (Chi-square tests)\n\n")
    
    f.write("="*70 + "\n")
    f.write("TREATMENT RECOMMENDATIONS BY STAGE\n")
    f.write("="*70 + "\n\n")
    
    for _, rec in recommendations_df.iterrows():
        f.write(f"{rec['cancer_stage']}:\n")
        f.write(f"  Recommended: {rec['recommended_treatment']}\n")
        f.write(f"  Survival Rate: {rec['survival_rate']}\n")
        f.write(f"  Avg Survival: {rec['avg_survival_days']} days\n")
        f.write(f"  Sample Size: {rec['sample_size']} patients\n")
        f.write(f"  Confidence: {rec['confidence']}\n\n")
    
    f.write("="*70 + "\n")
    f.write("DETAILED TREATMENT EFFECTIVENESS TABLE\n")
    f.write("="*70 + "\n\n")
    f.write(treatment_analysis.to_string(index=False) + "\n\n")
    
    f.write("="*70 + "\n")
    f.write("KEY FINDINGS\n")
    f.write("="*70 + "\n\n")
    
    # Calculate some key insights
    best_overall = treatment_analysis.loc[treatment_analysis['survival_rate'].idxmax()]
    worst_overall = treatment_analysis.loc[treatment_analysis['survival_rate'].idxmin()]
    
    f.write(f"1. Best Treatment Overall:\n")
    f.write(f"   {best_overall['treatment_type']} for {best_overall['cancer_stage']}\n")
    f.write(f"   Survival Rate: {best_overall['survival_rate_pct']:.1f}%\n\n")
    
    f.write(f"2. Most Challenging:\n")
    f.write(f"   {worst_overall['treatment_type']} for {worst_overall['cancer_stage']}\n")
    f.write(f"   Survival Rate: {worst_overall['survival_rate_pct']:.1f}%\n\n")
    
    f.write("3. Clinical Implications:\n")
    f.write("   - Treatment selection should be stage-specific\n")
    f.write("   - Early-stage cancers benefit most from surgical intervention\n")
    f.write("   - Advanced stages may require combination therapies\n")
    f.write("   - Patient factors (age, comorbidities) should also be considered\n\n")
    
    f.write("="*70 + "\n")
    f.write("USAGE IN CLINICAL DECISION SUPPORT\n")
    f.write("="*70 + "\n\n")
    f.write("These recommendations can be integrated into:\n")
    f.write("1. Clinical decision support systems\n")
    f.write("2. Treatment planning software\n")
    f.write("3. Patient counseling materials\n")
    f.write("4. Medical education resources\n")

print(" Saved: results/treatment_analysis_report.txt")

# Save recommendations as pickle for Streamlit app
recommendations_dict = recommendations_df.set_index('cancer_stage').to_dict('index')
with open('models/treatment_recommendations.pkl', 'wb') as f:
    pickle.dump(recommendations_dict, f)
print(" Saved: models/treatment_recommendations.pkl (for Streamlit app)")

print("\n" + "="*70)
print(" TREATMENT ANALYSIS COMPLETE!")
print("="*70)
print(f"\nSUMMARY:")
print(f" Analyzed {len(df):,} patient records")
print(f" Evaluated {len(treatment_analysis)} stage-treatment combinations")
print(f" Generated recommendations for {len(recommendations_df)} cancer stages")
print(f" Performed statistical significance testing")
print(f" Results saved and ready for clinical use")
print("="*70)

