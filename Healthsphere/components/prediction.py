import joblib
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

class DiseasePredictor:
    def __init__(self):
        """Initialize the disease predictor with models and feature information."""
        self.models = {}
        self.scalers = {}
        
        # Define feature information for each disease type
        self.disease_info = {
            'heart': {
                'features': ['Age', 'Gender', 'Chest Pain Type', 'Resting Blood Pressure', 'Serum Cholesterol (mg/dl)', 
                           'Fasting Blood Sugar > 120 mg/dl', 'Resting Electrocardiographic Results', 
                           'Maximum Heart Rate Achieved', 'Exercise Induced Angina', 'ST Depression Induced by Exercise', 
                           'Slope of the Peak Exercise ST Segment', 'Number of Major Vessels Colored by Fluoroscopy', 'Thalassemia'],
                'types': {
                    'Age': 'numeric',
                    'Gender': 'categorical',
                    'Chest Pain Type': 'categorical',
                    'Resting Blood Pressure': 'numeric',
                    'Serum Cholesterol (mg/dl)': 'numeric',
                    'Fasting Blood Sugar > 120 mg/dl': 'categorical',
                    'Resting Electrocardiographic Results': 'categorical',
                    'Maximum Heart Rate Achieved': 'numeric',
                    'Exercise Induced Angina': 'categorical',
                    'ST Depression Induced by Exercise': 'numeric',
                    'Slope of the Peak Exercise ST Segment': 'categorical',
                    'Number of Major Vessels Colored by Fluoroscopy': 'numeric',
                    'Thalassemia': 'categorical'
                },
                'categories': {
                    'Gender': ['Female', 'Male'],
                    'Chest Pain Type': ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'],
                    'Fasting Blood Sugar > 120 mg/dl': ['No', 'Yes'],
                    'Resting Electrocardiographic Results': ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'],
                    'Exercise Induced Angina': ['No', 'Yes'],
                    'Slope of the Peak Exercise ST Segment': ['Upsloping', 'Flat', 'Downsloping'],
                    'Thalassemia': ['Normal', 'Fixed Defect', 'Reversible Defect', 'Unknown']
                },
                'ranges': {
                    'Age': '0-120',
                    'Resting Blood Pressure': '50-250',
                    'Serum Cholesterol (mg/dl)': '100-600',
                    'Maximum Heart Rate Achieved': '50-250',
                    'ST Depression Induced by Exercise': '0-10',
                    'Number of Major Vessels Colored by Fluoroscopy': '0-4'
                },
                'labels': {
                    'Age': 'Age (years)',
                    'Gender': 'Gender',
                    'Chest Pain Type': 'Chest Pain Type',
                    'Resting Blood Pressure': 'Resting Blood Pressure (mm Hg)',
                    'Serum Cholesterol (mg/dl)': 'Serum Cholesterol (mg/dl)',
                    'Fasting Blood Sugar > 120 mg/dl': 'Fasting Blood Sugar > 120 mg/dl',
                    'Resting Electrocardiographic Results': 'Resting Electrocardiographic Results',
                    'Maximum Heart Rate Achieved': 'Maximum Heart Rate Achieved',
                    'Exercise Induced Angina': 'Exercise Induced Angina',
                    'ST Depression Induced by Exercise': 'ST Depression Induced by Exercise',
                    'Slope of the Peak Exercise ST Segment': 'Slope of Peak Exercise ST Segment',
                    'Number of Major Vessels Colored by Fluoroscopy': 'Number of Major Vessels Colored by Fluoroscopy',
                    'Thalassemia': 'Thalassemia Type'
                }
            },
            'diabetes': {
                'features': ['Pregnancies', 'Glucose Level', 'Blood Pressure', 'Skin Thickness', 
                           'Insulin Level', 'Body Mass Index', 'Diabetes Pedigree Function', 'Age'],
                'types': {
                    'Pregnancies': 'numeric',
                    'Glucose Level': 'numeric',
                    'Blood Pressure': 'numeric',
                    'Skin Thickness': 'numeric',
                    'Insulin Level': 'numeric',
                    'Body Mass Index': 'numeric',
                    'Diabetes Pedigree Function': 'numeric',
                    'Age': 'numeric'
                },
                'ranges': {
                    'Pregnancies': '0-20',
                    'Glucose Level': '0-200',
                    'Blood Pressure': '0-122',
                    'Skin Thickness': '0-100',
                    'Insulin Level': '0-846',
                    'Body Mass Index': '0-67.1',
                    'Diabetes Pedigree Function': '0.078-2.42',
                    'Age': '21-81'
                },
                'labels': {
                    'Pregnancies': 'Number of Pregnancies',
                    'Glucose Level': 'Blood Glucose Level (mg/dL)',
                    'Blood Pressure': 'Blood Pressure (mm Hg)',
                    'Skin Thickness': 'Skin Thickness (mm)',
                    'Insulin Level': 'Insulin Level (mu U/ml)',
                    'Body Mass Index': 'Body Mass Index (kg/m²)',
                    'Diabetes Pedigree Function': 'Diabetes Pedigree Function',
                    'Age': 'Age (years)'
                }
            },
            'stroke': {
                'features': ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                           'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'],
                'types': {
                    'gender': 'categorical',
                    'age': 'numeric',
                    'hypertension': 'categorical',
                    'heart_disease': 'categorical',
                    'ever_married': 'categorical',
                    'work_type': 'categorical',
                    'Residence_type': 'categorical',
                    'avg_glucose_level': 'numeric',
                    'bmi': 'numeric',
                    'smoking_status': 'categorical'
                },
                'categories': {
                    'gender': ['Male', 'Female', 'Other'],
                    'hypertension': ['No', 'Yes'],
                    'heart_disease': ['No', 'Yes'],
                    'ever_married': ['No', 'Yes'],
                    'work_type': ['Private', 'Self-employed', 'Government', 'Never worked', 'Children'],
                    'Residence_type': ['Urban', 'Rural'],
                    'smoking_status': ['Never smoked', 'Formerly smoked', 'Smokes', 'Unknown']
                },
                'ranges': {
                    'age': '0-120',
                    'avg_glucose_level': '50-400',
                    'bmi': '10-70'
                },
                'labels': {
                    'gender': 'Gender',
                    'age': 'Age (years)',
                    'hypertension': 'Has Hypertension',
                    'heart_disease': 'Has Heart Disease',
                    'ever_married': 'Ever Married',
                    'work_type': 'Type of Work',
                    'Residence_type': 'Type of Residence',
                    'avg_glucose_level': 'Average Glucose Level (mg/dL)',
                    'bmi': 'Body Mass Index (kg/m²)',
                    'smoking_status': 'Smoking Status'
                }
            },
            'kidney': {
                'features': ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
                           'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
                           'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
                           'potassium', 'hemoglobin', 'packed_cell_volume', 'white_blood_cell_count',
                           'red_blood_cell_count', 'hypertension', 'diabetes_mellitus',
                           'coronary_artery_disease', 'appetite', 'peda_edema', 'aanemia'],
                'types': {
                    'age': 'numeric',
                    'blood_pressure': 'numeric',
                    'specific_gravity': 'numeric',
                    'albumin': 'categorical',
                    'sugar': 'categorical',
                    'red_blood_cells': 'categorical',
                    'pus_cell': 'categorical',
                    'pus_cell_clumps': 'categorical',
                    'bacteria': 'categorical',
                    'blood_glucose_random': 'numeric',
                    'blood_urea': 'numeric',
                    'serum_creatinine': 'numeric',
                    'sodium': 'numeric',
                    'potassium': 'numeric',
                    'hemoglobin': 'numeric',
                    'packed_cell_volume': 'numeric',
                    'white_blood_cell_count': 'numeric',
                    'red_blood_cell_count': 'numeric',
                    'hypertension': 'categorical',
                    'diabetes_mellitus': 'categorical',
                    'coronary_artery_disease': 'categorical',
                    'appetite': 'categorical',
                    'peda_edema': 'categorical',
                    'aanemia': 'categorical'
                },
                'categories': {
                    'albumin': ['0', '1', '2', '3', '4', '5'],
                    'sugar': ['0', '1', '2', '3', '4', '5'],
                    'red_blood_cells': ['Normal', 'Abnormal'],
                    'pus_cell': ['Normal', 'Abnormal'],
                    'pus_cell_clumps': ['Present', 'Not Present'],
                    'bacteria': ['Present', 'Not Present'],
                    'hypertension': ['Yes', 'No'],
                    'diabetes_mellitus': ['Yes', 'No'],
                    'coronary_artery_disease': ['Yes', 'No'],
                    'appetite': ['Good', 'Poor'],
                    'peda_edema': ['Yes', 'No'],
                    'aanemia': ['Yes', 'No']
                },
                'ranges': {
                    'age': '2-90',
                    'blood_pressure': '50-180',
                    'specific_gravity': '1.005-1.025',
                    'blood_glucose_random': '22-490',
                    'blood_urea': '1.5-391',
                    'serum_creatinine': '0.4-15.2',
                    'sodium': '4.5-163',
                    'potassium': '2.5-47',
                    'hemoglobin': '3.1-17.8',
                    'packed_cell_volume': '9-54',
                    'white_blood_cell_count': '2200-26400',
                    'red_blood_cell_count': '2.1-8'
                },
                'labels': {
                    'age': 'Age (years)',
                    'blood_pressure': 'Blood Pressure (mm Hg)',
                    'specific_gravity': 'Specific Gravity',
                    'albumin': 'Albumin Level',
                    'sugar': 'Sugar Level',
                    'red_blood_cells': 'Red Blood Cells',
                    'pus_cell': 'Pus Cell',
                    'pus_cell_clumps': 'Pus Cell Clumps',
                    'bacteria': 'Bacteria',
                    'blood_glucose_random': 'Blood Glucose Random (mg/dL)',
                    'blood_urea': 'Blood Urea (mg/dL)',
                    'serum_creatinine': 'Serum Creatinine (mg/dL)',
                    'sodium': 'Sodium (mEq/L)',
                    'potassium': 'Potassium (mEq/L)',
                    'hemoglobin': 'Hemoglobin (g/dL)',
                    'packed_cell_volume': 'Packed Cell Volume (%)',
                    'white_blood_cell_count': 'White Blood Cell Count (cells/mm³)',
                    'red_blood_cell_count': 'Red Blood Cell Count (cells/mm³)',
                    'hypertension': 'Hypertension',
                    'diabetes_mellitus': 'Diabetes Mellitus',
                    'coronary_artery_disease': 'Coronary Artery Disease',
                    'appetite': 'Appetite',
                    'peda_edema': 'Peda Edema',
                    'aanemia': 'Aanemia'
                }
            },
            'liver': {
                'features': ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
                           'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
                           'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
                           'Albumin_Globulin_Ratio'],
                'types': {
                    'Age': 'numeric',
                    'Gender': 'categorical',
                    'Total_Bilirubin': 'numeric',
                    'Direct_Bilirubin': 'numeric',
                    'Alkaline_Phosphotase': 'numeric',
                    'Alamine_Aminotransferase': 'numeric',
                    'Aspartate_Aminotransferase': 'numeric',
                    'Total_Protiens': 'numeric',
                    'Albumin': 'numeric',
                    'Albumin_Globulin_Ratio': 'numeric'
                },
                'categories': {
                    'Gender': ['Female', 'Male']
                },
                'ranges': {
                    'Age': '4-90',
                    'Total_Bilirubin': '0.4-75',
                    'Direct_Bilirubin': '0.1-19.7',
                    'Alkaline_Phosphotase': '63-2110',
                    'Alamine_Aminotransferase': '10-2000',
                    'Aspartate_Aminotransferase': '10-4929',
                    'Total_Protiens': '2.7-9.6',
                    'Albumin': '1.0-5.5',
                    'Albumin_Globulin_Ratio': '0.3-2.8'
                },
                'labels': {
                    'Age': 'Age (years)',
                    'Gender': 'Gender',
                    'Total_Bilirubin': 'Total Bilirubin (mg/dL)',
                    'Direct_Bilirubin': 'Direct Bilirubin (mg/dL)',
                    'Alkaline_Phosphotase': 'Alkaline Phosphotase (U/L)',
                    'Alamine_Aminotransferase': 'Alamine Aminotransferase (U/L)',
                    'Aspartate_Aminotransferase': 'Aspartate Aminotransferase (U/L)',
                    'Total_Protiens': 'Total Proteins (g/dL)',
                    'Albumin': 'Albumin (g/dL)',
                    'Albumin_Globulin_Ratio': 'Albumin Globulin Ratio'
                }
            },
            'parkinsons': {
                'features': ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                           'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                           'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                           'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
                           'spread1', 'spread2', 'D2', 'PPE'],
                'types': {
                    'MDVP:Fo(Hz)': 'numeric',
                    'MDVP:Fhi(Hz)': 'numeric',
                    'MDVP:Flo(Hz)': 'numeric',
                    'MDVP:Jitter(%)': 'numeric',
                    'MDVP:Jitter(Abs)': 'numeric',
                    'MDVP:RAP': 'numeric',
                    'MDVP:PPQ': 'numeric',
                    'Jitter:DDP': 'numeric',
                    'MDVP:Shimmer': 'numeric',
                    'MDVP:Shimmer(dB)': 'numeric',
                    'Shimmer:APQ3': 'numeric',
                    'Shimmer:APQ5': 'numeric',
                    'MDVP:APQ': 'numeric',
                    'Shimmer:DDA': 'numeric',
                    'NHR': 'numeric',
                    'HNR': 'numeric',
                    'RPDE': 'numeric',
                    'DFA': 'numeric',
                    'spread1': 'numeric',
                    'spread2': 'numeric',
                    'D2': 'numeric',
                    'PPE': 'numeric'
                },
                'ranges': {
                    'MDVP:Fo(Hz)': '88.333-260.105',
                    'MDVP:Fhi(Hz)': '102.145-592.03',
                    'MDVP:Flo(Hz)': '65.476-239.17',
                    'MDVP:Jitter(%)': '0.00168-0.03316',
                    'MDVP:Jitter(Abs)': '0.000007-0.00026',
                    'MDVP:RAP': '0.00068-0.02144',
                    'MDVP:PPQ': '0.00092-0.01958',
                    'Jitter:DDP': '0.00204-0.06433',
                    'MDVP:Shimmer': '0.00954-0.11908',
                    'MDVP:Shimmer(dB)': '0.085-1.302',
                    'Shimmer:APQ3': '0.00455-0.05647',
                    'Shimmer:APQ5': '0.0057-0.07968',
                    'MDVP:APQ': '0.00719-0.13778',
                    'Shimmer:DDA': '0.01367-0.16942',
                    'NHR': '0.000650-0.314783',
                    'HNR': '8.441-33.047',
                    'RPDE': '0.256570-0.685151',
                    'DFA': '0.574282-0.825288',
                    'spread1': '-7.964984-13.295',
                    'spread2': '0.006274-0.450493',
                    'D2': '1.423287-3.671155',
                    'PPE': '0.044539-0.527367'
                },
                'labels': {
                    'MDVP:Fo(Hz)': 'MDVP Fo (Hz)',
                    'MDVP:Fhi(Hz)': 'MDVP Fhi (Hz)',
                    'MDVP:Flo(Hz)': 'MDVP Flo (Hz)',
                    'MDVP:Jitter(%)': 'MDVP Jitter (%)',
                    'MDVP:Jitter(Abs)': 'MDVP Jitter (Abs)',
                    'MDVP:RAP': 'MDVP RAP',
                    'MDVP:PPQ': 'MDVP PPQ',
                    'Jitter:DDP': 'Jitter DDP',
                    'MDVP:Shimmer': 'MDVP Shimmer',
                    'MDVP:Shimmer(dB)': 'MDVP Shimmer (dB)',
                    'Shimmer:APQ3': 'Shimmer APQ3',
                    'Shimmer:APQ5': 'Shimmer APQ5',
                    'MDVP:APQ': 'MDVP APQ',
                    'Shimmer:DDA': 'Shimmer DDA',
                    'NHR': 'NHR',
                    'HNR': 'HNR',
                    'RPDE': 'RPDE',
                    'DFA': 'DFA',
                    'spread1': 'Spread 1',
                    'spread2': 'Spread 2',
                    'D2': 'D2',
                    'PPE': 'PPE'
                }
            }
        }
        
        # Define value mappings for categorical variables
        self.value_mappings = {
            'heart': {
                'Gender': {'Female': 0, 'Male': 1},
                'Chest Pain Type': {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3},
                'Fasting Blood Sugar > 120 mg/dl': {'No': 0, 'Yes': 1},
                'Resting Electrocardiographic Results': {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2},
                'Exercise Induced Angina': {'No': 0, 'Yes': 1},
                'Slope of the Peak Exercise ST Segment': {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2},
                'Thalassemia': {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2, 'Unknown': 3}
            },
            'stroke': {
                'gender': {'Male': 0, 'Female': 1, 'Other': 2},
                'hypertension': {'No': 0, 'Yes': 1},
                'heart_disease': {'No': 0, 'Yes': 1},
                'ever_married': {'No': 0, 'Yes': 1},
                'work_type': {'Private': 0, 'Self-employed': 1, 'Government': 2, 'Never worked': 3, 'Children': 4},
                'Residence_type': {'Urban': 0, 'Rural': 1},
                'smoking_status': {'Never smoked': 0, 'Formerly smoked': 1, 'Smokes': 2, 'Unknown': 3}
            },
            'kidney': {
                'red_blood_cells': {'Normal': 0, 'Abnormal': 1},
                'pus_cell': {'Normal': 0, 'Abnormal': 1},
                'pus_cell_clumps': {'Present': 1, 'Not Present': 0},
                'bacteria': {'Present': 1, 'Not Present': 0},
                'hypertension': {'Yes': 1, 'No': 0},
                'diabetes_mellitus': {'Yes': 1, 'No': 0},
                'coronary_artery_disease': {'Yes': 1, 'No': 0},
                'appetite': {'Good': 1, 'Poor': 0},
                'peda_edema': {'Yes': 1, 'No': 0},
                'aanemia': {'Yes': 1, 'No': 0}
            },
            'liver': {
                'Gender': {'Female': 0, 'Male': 1}
            }
        }
        
        # Define reverse mappings for PDF data conversion
        self.reverse_mappings = {
            'heart': {
                'Gender': {0: 'Female', 1: 'Male'},
                'Chest Pain Type': {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'},
                'Fasting Blood Sugar > 120 mg/dl': {0: 'No', 1: 'Yes'},
                'Resting Electrocardiographic Results': {0: 'Normal', 1: 'ST-T Wave Abnormality', 2: 'Left Ventricular Hypertrophy'},
                'Exercise Induced Angina': {0: 'No', 1: 'Yes'},
                'Slope of the Peak Exercise ST Segment': {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'},
                'Thalassemia': {0: 'Normal', 1: 'Fixed Defect', 2: 'Reversible Defect', 3: 'Unknown'}
            },
            'stroke': {
                'gender': {0: 'Male', 1: 'Female', 2: 'Other'},
                'hypertension': {0: 'No', 1: 'Yes'},
                'heart_disease': {0: 'No', 1: 'Yes'},
                'ever_married': {0: 'No', 1: 'Yes'},
                'work_type': {0: 'Private', 1: 'Self-employed', 2: 'Government', 3: 'Never worked', 4: 'Children'},
                'Residence_type': {0: 'Urban', 1: 'Rural'},
                'smoking_status': {0: 'Never smoked', 1: 'Formerly smoked', 2: 'Smokes', 3: 'Unknown'}
            },
            'kidney': {
                'red_blood_cells': {0: 'Normal', 1: 'Abnormal'},
                'pus_cell': {0: 'Normal', 1: 'Abnormal'},
                'pus_cell_clumps': {0: 'Not Present', 1: 'Present'},
                'bacteria': {0: 'Not Present', 1: 'Present'},
                'hypertension': {0: 'No', 1: 'Yes'},
                'diabetes_mellitus': {0: 'No', 1: 'Yes'},
                'coronary_artery_disease': {0: 'No', 1: 'Yes'},
                'appetite': {0: 'Poor', 1: 'Good'},
                'peda_edema': {0: 'No', 1: 'Yes'},
                'aanemia': {0: 'No', 1: 'Yes'}
            },
            'liver': {
                'Gender': {0: 'Female', 1: 'Male'}
            }
        }
        
        # Load models and scalers
        self.load_models()
    
    def load_models(self):
        # Get the absolute path to models directory
        # Models directory is at the same level as Healthsphere directory
        project_root = os.path.dirname(os.path.dirname(__file__))  # Healthsphere directory
        parent_dir = os.path.dirname(project_root)                 # Test directory
        models_dir = os.path.join(parent_dir, 'models')            # Test/models directory
        
        print(f"Project root: {project_root}")
        print(f"Parent directory: {parent_dir}")
        print(f"Models directory: {models_dir}")
        print(f"Models directory exists: {os.path.exists(models_dir)}")
        
        # Load traditional ML models
        for disease in ['diabetes', 'heart', 'stroke', 'kidney', 'liver', 'parkinsons']:
            try:
                model_path = os.path.join(models_dir, f'{disease}_model.joblib')
                scaler_path = os.path.join(models_dir, f'{disease}_scaler.joblib')
                features_path = os.path.join(models_dir, f'{disease}_features.txt')
                
                # Load model and scaler
                self.models[disease] = joblib.load(model_path)
                self.scalers[disease] = joblib.load(scaler_path)
                
                # Load feature list if available
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        features = f.read().strip().split(',')
                    # Update features list if different from current definition
                    if set(features) != set(self.disease_info[disease]['features']):
                        print(f"Updating {disease} features from trained model")
                        self.disease_info[disease]['features'] = features
                
                print(f"Successfully loaded {disease} model from {model_path}")
            except Exception as e:
                print(f"Error loading {disease} model: {str(e)}")
                # Try alternative model locations
                try:
                    alt_model_path = os.path.join(project_root, 'models', f'{disease}_model.joblib')
                    alt_scaler_path = os.path.join(project_root, 'models', f'{disease}_scaler.joblib')
                    
                    if os.path.exists(alt_model_path) and os.path.exists(alt_scaler_path):
                        self.models[disease] = joblib.load(alt_model_path)
                        self.scalers[disease] = joblib.load(alt_scaler_path)
                        print(f"Successfully loaded {disease} model from alternative path: {alt_model_path}")
                except Exception as alt_e:
                    print(f"Error loading {disease} model from alternative path: {str(alt_e)}")
    
    def get_feature_info(self, disease_type):
        """Get feature information for specified disease type"""
        return self.disease_info.get(disease_type, {})
    
    def check_missing_features(self, disease_type, input_data):
        """Check which required features are missing from input data"""
        required = set(self.disease_info[disease_type]['features'])
        provided = set(input_data.keys())
        return list(required - provided)
    
    def validate_input(self, disease_type, input_data):
        """Validate input data against defined ranges and categories"""
        disease = self.disease_info[disease_type]
        errors = []
        
        for feature, value in input_data.items():
            if feature not in disease['features']:
                continue
                
            feature_type = disease['types'][feature]
            
            if feature_type == 'numeric':
                try:
                    value = float(value)
                    if 'ranges' in disease and feature in disease['ranges']:
                        min_val, max_val = map(float, disease['ranges'][feature].split('-'))
                        if value < min_val or value > max_val:
                            errors.append(f"{feature} should be between {min_val} and {max_val}")
                except ValueError:
                    errors.append(f"{feature} should be a number")
            
            elif feature_type == 'categorical':
                if str(value) not in disease['categories'][feature]:
                    errors.append(f"{feature} should be one of {disease['categories'][feature]}")
        
        return len(errors) == 0, errors
    
    def convert_pdf_data(self, disease_type, extracted_data):
        """Convert numeric values from PDF to user-friendly text values"""
        converted_data = {}
        feature_info = self.disease_info.get(disease_type, {})
        
        for feature, value in extracted_data.items():
            feature_type = feature_info['types'].get(feature)
            
            if feature_type == 'categorical':
                # Try to convert string number to int first
                try:
                    numeric_value = int(float(value))
                    # Convert numeric value to text using reverse mapping
                    if (disease_type in self.reverse_mappings and 
                        feature in self.reverse_mappings[disease_type] and 
                        numeric_value in self.reverse_mappings[disease_type][feature]):
                        converted_data[feature] = self.reverse_mappings[disease_type][feature][numeric_value]
                    else:
                        converted_data[feature] = value
                except (ValueError, TypeError):
                    # If value is already text, keep it as is
                    converted_data[feature] = value
            else:
                # For numeric fields, keep the value as is
                converted_data[feature] = value
        
        return converted_data
    
    def categorical_to_numeric(self, disease_type, feature, value):
        """Convert categorical text value to numeric value for model input"""
        if disease_type in self.value_mappings and feature in self.value_mappings[disease_type]:
            mapping = self.value_mappings[disease_type]
            if value in mapping[feature]:
                return mapping[feature][value]
        
        # If we can't find a mapping, try to convert directly
        try:
            return float(value)
        except (ValueError, TypeError):
            # If all else fails, return the original value
            return value
    
    def predict(self, disease_type, input_data):
        """Make prediction"""
        try:
            if disease_type not in self.models:
                return None, f"No model found for {disease_type}"
            
            # Get feature information
            feature_info = self.get_feature_info(disease_type)
            if not feature_info:
                return None, f"No feature information found for {disease_type}"
            
            features = feature_info.get('features', [])
            if not features:
                return None, "No features defined for this disease type"
            
            # Validate input data
            validation_errors = []
            processed_data = {}
            
            for feature in features:
                if feature not in input_data:
                    validation_errors.append(f"Missing feature: {feature}")
                    continue
                
                feature_type = feature_info['types'].get(feature, 'numeric')  # Default to numeric
                value = input_data[feature]
                
                try:
                    if feature_type == 'categorical':
                        # Try categorical mapping
                        if disease_type in self.value_mappings and feature in self.value_mappings[disease_type]:
                            if value in self.value_mappings[disease_type][feature]:
                                processed_data[feature] = self.value_mappings[disease_type][feature][value]
                            else:
                                # Try to convert numeric value
                                try:
                                    numeric_value = float(value)
                                    processed_data[feature] = numeric_value
                                except (ValueError, TypeError):
                                    validation_errors.append(f"Invalid value for {feature}: {value}")
                        else:
                            # If no mapping exists, try to convert to float
                            try:
                                processed_data[feature] = float(value)
                            except (ValueError, TypeError):
                                validation_errors.append(f"Invalid value for {feature}: {value}")
                    else:
                        # For numeric features, convert to float
                        try:
                            processed_data[feature] = float(value)
                        except (ValueError, TypeError):
                            validation_errors.append(f"Invalid value for {feature}: {value}")
                except Exception as e:
                    validation_errors.append(f"Error processing {feature}: {str(e)}")
            
            if validation_errors:
                return None, f"Validation errors: {', '.join(validation_errors)}"
            
            # Handle missing features with default values
            missing_features = set(features) - set(processed_data.keys())
            if missing_features:
                for feature in missing_features:
                    # Use mean value for missing numeric features
                    processed_data[feature] = 0.0
                print(f"Using default values for missing features: {missing_features}")
            
            # Convert to pandas DataFrame with features in correct order
            X = pd.DataFrame([[processed_data.get(f, 0.0) for f in features]], columns=features)
            
            # Scale the features
            X_scaled = self.scalers[disease_type].transform(X)
            
            # Make prediction
            prediction = self.models[disease_type].predict(X_scaled)
            prediction_proba = self.models[disease_type].predict_proba(X_scaled)
            
            result = {
                'prediction': bool(prediction[0]),
                'prediction_probability': float(prediction_proba[0][1]),
                'confidence': float(prediction_proba[0][1]) if prediction[0] else float(prediction_proba[0][0]),
                'input_data': processed_data
            }
            
            return result, None
            
        except Exception as e:
            import traceback
            print(f"Error in predict: {str(e)}")
            print(traceback.format_exc())
            return None, str(e)