# Cancer Survival Prediction System

A machine learning-powered clinical decision support system for cancer patient survival prediction, treatment recommendations, and risk assessment.

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r streamlit_app/requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run streamlit_app/app.py
   ```
   
   **Note**: Run this command from the project root directory (`Cancer-Survival-Prediction-System`), not from inside the `streamlit_app` folder.

3. **Access the Application**
   Open your browser and navigate to `http://localhost:8501`

## Features

- **Survival Predictor**: Predict patient survival probability and time estimation
- **Cancer Risk Assessment**: Environmental and lifestyle-based risk prediction
- **Treatment Recommender**: Evidence-based treatment suggestions by cancer stage
- **Model Performance Dashboard**: Comprehensive metrics and model comparisons
- **Data Explorer**: Interactive visualization of dataset insights

## Project Structure

```
├── data/                       # Processed datasets and encoders
├── models/                     # Trained ML models (.pkl files)
├── results/                    # Analysis outputs and visualizations
├── src/                        # Training and analysis scripts
│   ├── 01_data_preprocessing.py
│   ├── 02_handle_imbalance.py
│   ├── 03_train_classification.py
│   ├── 05_train_regression.py
│   ├── 06_treatment_analysis.py
│   ├── 10_advanced_visualizations.py
│   ├── 11_cox_survival_model.py
│   └── 13_cancer_risk_prediction.py
└── streamlit_app/             # Web application
    ├── app.py                 # Main Streamlit app
    └── requirements.txt       # Python dependencies
```

## Models

- **Classification Model**: XGBoost (43.5% recall, 67.4% ROC-AUC)
- **Regression Model**: Linear Regression (±110 days average error)
- **Risk Predictor**: Multi-class classifier for cancer risk levels
- **Treatment Analysis**: Statistical analysis for treatment effectiveness

## Training Pipeline (Optional)

If you want to retrain models from scratch:

1. **Data Preprocessing**
   ```bash
   python src/01_data_preprocessing.py
   ```

2. **Handle Class Imbalance**
   ```bash
   python src/02_handle_imbalance.py
   ```

3. **Train Classification Model**
   ```bash
   python src/03_train_classification.py
   ```

4. **Train Regression Model**
   ```bash
   python src/05_train_regression.py
   ```

5. **Train Risk Predictor**
   ```bash
   python src/13_cancer_risk_prediction.py
   ```

## Data Requirements

- **Lung Cancer Dataset**: 100,000 patient records with demographics, clinical features, and survival outcomes
- **Environmental Dataset**: 1,000 records with air pollution and lifestyle factors for risk prediction

## Clinical Use Cases

- **Patient Triage**: Risk stratification and priority assignment
- **Treatment Planning**: Duration estimation and resource allocation
- **Clinical Decision Support**: Evidence-based treatment recommendations
- **Patient Counseling**: Prognosis communication and expectation setting

## Disclaimer

This system is designed to support, not replace, clinical judgment. All predictions should be validated by qualified healthcare professionals and used in conjunction with comprehensive patient assessment.
