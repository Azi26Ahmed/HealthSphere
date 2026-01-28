import pandas as pd
import numpy as np
import streamlit as st
from components.utils import safe_parse_datetime

class UtilsVisualizer:
    """Utility functions for visualization."""
    
    def generate_synthetic_data(self, n_samples=100):
        """Generate synthetic health data for demonstration purposes."""
        return generate_synthetic_data(n_samples)
    
    def get_disease_parameters(self, disease_type):
        """Get parameter names for a specific disease type."""
        return get_disease_parameters(disease_type)
    
    def format_parameter_name(self, param_name):
        """Format parameter names for better display."""
        return format_parameter_name(param_name)
    
    def create_correlation_matrix(self, param_data, timestamps):
        """Create a correlation matrix from parameter data."""
        return create_correlation_matrix(param_data, timestamps)
    
    def safe_parse_datetime_viz(self, timestamp_str):
        """Safe datetime parsing specifically for visualization."""
        return safe_parse_datetime_viz(timestamp_str)
    
    def get_color_palette(self, theme="primary"):
        """Get color palette for visualizations."""
        return get_color_palette(theme)
    
    def format_disease_name(self, disease_key):
        """Format disease key to display name."""
        return format_disease_name(disease_key)

def generate_synthetic_data(n_samples=100):
    """Generate synthetic health data for demonstration purposes."""
    np.random.seed(42)
    
    # Generate synthetic parameters for different diseases
    data = {
        'age': np.random.randint(20, 80, n_samples),
        'bmi': np.random.normal(25, 5, n_samples),
        'blood_pressure_systolic': np.random.normal(120, 20, n_samples),
        'blood_pressure_diastolic': np.random.normal(80, 10, n_samples),
        'cholesterol': np.random.normal(200, 40, n_samples),
        'glucose': np.random.normal(100, 20, n_samples),
        'heart_rate': np.random.normal(70, 10, n_samples),
        'risk_score': np.random.uniform(0, 1, n_samples)
    }
    
    # Add categorical variables
    data['gender'] = np.random.choice(['Male', 'Female'], n_samples)
    data['smoking'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    data['exercise'] = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
    
    return pd.DataFrame(data)

def get_disease_parameters(disease_type):
    """Get parameter names for a specific disease type."""
    disease_params = {
        'heart': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
        'diabetes': ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree', 'age'],
        'liver': ['age', 'gender', 'total_bilirubin', 'direct_bilirubin', 'alkaline_phosphotase', 'alamine_aminotransferase', 'aspartate_aminotransferase', 'total_proteins', 'albumin', 'albumin_and_globulin_ratio'],
        'kidney': ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium', 'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count', 'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema', 'aanemia'],
        'stroke': ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'],
        'parkinsons': ['mdvp_fo', 'mdvp_fhi', 'mdvp_flo', 'mdvp_jitter_percent', 'mdvp_jitter_abs', 'mdvp_rap', 'mdvp_ppq', 'jitter_ddp', 'mdvp_shimmer', 'mdvp_shimmer_db', 'shimmer_apq3', 'shimmer_apq5', 'mdvp_apq', 'shimmer_dda', 'nhr', 'hnr', 'rpde', 'dfa', 'd2', 'ppe'],
        'brain_tumor': ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness', 'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension']
    }
    return disease_params.get(disease_type, [])

def format_parameter_name(param_name):
    """Format parameter names for better display."""
    # Replace underscores with spaces and capitalize
    formatted = param_name.replace('_', ' ').title()
    
    # Handle special cases
    replacements = {
        'Bmi': 'BMI',
        'Mdvp': 'MDVP',
        'Nhr': 'NHR',
        'Hnr': 'HNR',
        'Rpde': 'RPDE',
        'Dfa': 'DFA',
        'Ppe': 'PPE',
        'Cp': 'CP',
        'Fbs': 'FBS',
        'Restecg': 'RestECG',
        'Thalach': 'Thalach',
        'Exang': 'ExAng',
        'Ca': 'CA',
        'Thal': 'Thal'
    }
    
    for old, new in replacements.items():
        formatted = formatted.replace(old, new)
    
    return formatted

def create_correlation_matrix(param_data, timestamps):
    """Create a correlation matrix from parameter data."""
    if not param_data or not timestamps:
        return None, None, None
    
    # Convert to DataFrame
    df_data = []
    all_params = set()
    
    for i, (params, timestamp) in enumerate(zip(param_data, timestamps)):
        if isinstance(params, dict):
            row = {'timestamp': timestamp, 'index': i}
            row.update(params)
            df_data.append(row)
            all_params.update(params.keys())
    
    if not df_data:
        return None, None, None
    
    df = pd.DataFrame(df_data)
    all_params = list(all_params)
    
    # Calculate correlation matrix for numeric columns only
    numeric_params = []
    for param in all_params:
        if param in df.columns:
            try:
                pd.to_numeric(df[param])
                numeric_params.append(param)
            except (ValueError, TypeError):
                continue
    
    if len(numeric_params) < 2:
        return None, None, None
    
    correlation_matrix = df[numeric_params].corr()
    
    return correlation_matrix, all_params, timestamps

def safe_parse_datetime_viz(timestamp_str):
    """Safe datetime parsing specifically for visualization."""
    return safe_parse_datetime(timestamp_str)

def get_color_palette(theme="primary"):
    """Get color palette for visualizations."""
    color_themes = {
        "primary": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
        "pastel": ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3", "#fdb462"],
        "diverging": ["#d53e4f", "#fc8d59", "#fee08b", "#e6f598", "#99d594", "#3288bd"]
    }
    return color_themes.get(theme, color_themes["primary"])

def format_disease_name(disease_key):
    """Format disease key to display name."""
    disease_options = {
        "heart": "Heart Disease",
        "diabetes": "Diabetes",
        "liver": "Liver Disease",
        "kidney": "Kidney Disease",
        "stroke": "Stroke",
        "parkinsons": "Parkinson's Disease",
        "brain_tumor": "Brain Tumor"
    }
    return disease_options.get(disease_key, disease_key.replace('_', ' ').title()) 