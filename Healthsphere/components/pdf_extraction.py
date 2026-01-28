import re
import os
import pdfplumber
from typing import Dict, List
import logging

# Define expected features for each disease
FEATURES_BY_DISEASE = {
    "diabetes": [
        "Pregnancies", "Glucose Level", "Blood Pressure", "Skin Thickness", "Insulin Level",
        "Body Mass Index", "Diabetes Pedigree Function", "Age"
    ],
    "heart": [
        "Age", "Gender", "Chest Pain Type", "Resting Blood Pressure", "Serum Cholesterol (mg/dl)",
        "Fasting Blood Sugar > 120 mg/dl", "Resting Electrocardiographic Results",
        "Maximum Heart Rate Achieved", "Exercise Induced Angina", "ST Depression Induced by Exercise",
        "Slope of the Peak Exercise ST Segment", "Number of Major Vessels Colored by Fluoroscopy",
        "Thalassemia"
    ],
    "stroke": [
        "Age", "Gender", "Chest Pain Type", "Resting Blood Pressure", "Serum Cholesterol (mg/dl)",
        "Fasting Blood Sugar > 120 mg/dl", "Resting Electrocardiographic Results",
        "Maximum Heart Rate Achieved", "Exercise Induced Angina", "ST Depression Induced by Exercise",
        "Slope of the Peak Exercise ST Segment", "Number of Major Vessels Colored by Fluoroscopy",
        "Thalassemia"
    ],
    "kidney": [
        "Age", "Blood Pressure", "Specific Gravity", "Albumin", "Sugar", "Red Blood Cells", "Pus Cell",
        "Pus Cell Clumps", "Bacteria", "Blood Glucose Random", "Blood Urea", "Serum Creatinine",
        "Sodium", "Potassium", "Hemoglobin", "Packed Cell Volume", "White Blood Cell Count",
        "Red Blood Cell Count", "Hypertension", "Diabetes Mellitus", "Coronary Artery Disease",
        "Appetite", "Pedal Edema", "Anemia"
    ],
    "liver": [
        "Age", "Gender", "Total Bilirubin", "Direct Bilirubin", "Alkaline Phosphotase",
        "Alanine Aminotransferase", "Aspartate Aminotransferase", "Total Proteins", "Albumin",
        "Albumin and Globulin Ratio"
    ],
    "parkinsons": [
        "Average Vocal Fundamental Frequency", "Maximum Vocal Fundamental Frequency",
        "Minimum Vocal Fundamental Frequency", "Jitter (%)", "Jitter (Abs)", "Relative Average Perturbation",
        "Pitch Period Perturbation Quotient", "DDP", "Shimmer", "Shimmer (dB)", "APQ3", "APQ5", "APQ",
        "DDA", "Noise-to-Harmonics Ratio", "Harmonics-to-Noise Ratio",
        "Recurrence Period Density Entropy", "Detrended Fluctuation Analysis", "Spread 1", "Spread 2",
        "D2", "Pitch Period Entropy"
    ]
}

# Categorical value mappings
CATEGORICAL_MAPPINGS = {
    "stroke": {
        "Gender": {"Male": 1, "Female": 0, "male": 1, "female": 0},
        "Ever Married": {"Yes": 1, "No": 0, "yes": 1, "no": 0},
        "Work Type": {
            "Private": 0, "Self-employed": 1, "Govt_job": 2, "Children": 3, "Never_worked": 4,
            "private": 0, "self-employed": 1, "govt_job": 2, "children": 3, "never_worked": 4
        },
        "Residence Type": {"Urban": 1, "Rural": 0, "urban": 1, "rural": 0},
        "Smoking Status": {
            "Never smoked": 0, "Formerly smoked": 1, "Smokes": 2, "Unknown": 3,
            "never smoked": 0, "formerly smoked": 1, "smokes": 2, "unknown": 3
        }
    },
    "kidney": {
        "Hypertension": {"yes": 1, "no": 0, "Yes": 1, "No": 0},
        "Diabetes Mellitus": {"yes": 1, "no": 0, "Yes": 1, "No": 0},
        "Coronary Artery Disease": {"yes": 1, "no": 0, "Yes": 1, "No": 0},
        "Appetite": {"good": 1, "poor": 0, "Good": 1, "Poor": 0},
        "Pedal Edema": {"yes": 1, "no": 0, "Yes": 1, "No": 0},
        "Anemia": {"yes": 1, "no": 0, "Yes": 1, "No": 0},
        "Pus Cell": {"normal": 0, "abnormal": 1, "Normal": 0, "Abnormal": 1},
        "Pus Cell Clumps": {"present": 1, "notpresent": 0, "Present": 1, "Notpresent": 0, "not present": 0, "Not present": 0},
        "Bacteria": {"present": 1, "notpresent": 0, "Present": 1, "Notpresent": 0, "not present": 0, "Not present": 0},
        "Red Blood Cells": {"normal": 0, "abnormal": 1, "Normal": 0, "Abnormal": 1}
    },
    "liver": {
        "Gender": {"Male": 1, "Female": 0, "male": 1, "female": 0}
    },
    "heart": {
        "Gender": {"Male": 1, "Female": 0, "male": 1, "female": 0}
    }
}

KEY_VALUE_REGEX = re.compile(r"^(.*?):\s*(.*)$")

# Add numeric_fields_by_disease dictionary definition
numeric_fields_by_disease = {
    'heart': ['Age', 'Resting Blood Pressure', 'Serum Cholesterol (mg/dl)', 'Maximum Heart Rate Achieved', 'ST Depression Induced by Exercise'],
    'stroke': ['Age', 'Average Glucose Level', 'Body Mass Index'],
    'liver': ['Age', 'Total Bilirubin', 'Direct Bilirubin', 'Alkaline Phosphotase', 'Alanine Aminotransferase', 
              'Aspartate Aminotransferase', 'Total Proteins', 'Albumin', 'Albumin and Globulin Ratio'],
    'kidney': ['Age', 'Blood Pressure', 'Specific Gravity', 'Albumin', 'Sugar', 'Blood Glucose Random', 'Blood Urea',
               'Serum Creatinine', 'Sodium', 'Potassium', 'Hemoglobin', 'Packed Cell Volume', 'White Blood Cell Count',
               'Red Blood Cell Count'],
    'diabetes': ['Pregnancies', 'Glucose Level', 'Blood Pressure', 'Skin Thickness', 'Insulin Level', 'Body Mass Index', 'Diabetes Pedigree Function', 'Age'],
    'parkinsons': ['Jitter (%)', 'Jitter (Abs)', 'Relative Average Perturbation', 'Pitch Period Perturbation Quotient', 
                  'DDP', 'Shimmer', 'Shimmer (dB)', 'APQ3', 'APQ5', 'APQ', 'DDA', 'Noise-to-Harmonics Ratio', 
                  'Harmonics-to-Noise Ratio', 'Recurrence Period Density Entropy', 'Detrended Fluctuation Analysis', 
                  'Spread 1', 'Spread 2', 'D2', 'Pitch Period Entropy']
}

# Define optional fields that shouldn't be validated as strictly
OPTIONAL_FIELDS = ["Patient ID", "Report Date", "Doctor Name", "Hospital", "Notes"]

# Define fields that should be explicitly cast to integers
INTEGER_FIELDS = {
    "heart": ["Gender", "Chest Pain Type", "Fasting Blood Sugar > 120 mg/dl", 
              "Resting Electrocardiographic Results", "Exercise Induced Angina",
              "Slope of the Peak Exercise ST Segment", "Number of Major Vessels Colored by Fluoroscopy",
              "Thalassemia"],
    "kidney": ["Red Blood Cells", "Pus Cell", "Pus Cell Clumps", "Bacteria", 
               "Hypertension", "Diabetes Mellitus", "Coronary Artery Disease",
               "Appetite", "Pedal Edema", "Anemia"],
    "stroke": ["Gender", "Hypertension", "Heart Disease", "Ever Married", 
               "Work Type", "Residence Type", "Smoking Status"],
    "liver": ["Gender"],
    "parkinsons": []
}

def safe_cast_float(val):
    """
    Safely cast a value to float, handling common issues like spaces and scientific notation.
    Returns None if conversion fails.
    """
    if val is None:
        return None
        
    try:
        # Handle string preprocessing
        if isinstance(val, str):
            # Remove spaces
            val = val.replace(" ", "")
            # Handle common scientific notation
            if 'e' in val.lower() or 'x10' in val.lower():
                # Try to standardize scientific notation
                if 'x10' in val.lower():
                    parts = val.lower().split('x10')
                    if len(parts) == 2:
                        try:
                            return float(parts[0]) * (10 ** float(parts[1]))
                        except:
                            pass
            
        # Try regular float conversion
        return float(val)
    except (ValueError, TypeError):
        return None

def get_typical_values(disease_type):
    """
    Return typical values for missing fields based on disease type.
    These values will be used as placeholders when a field is missing.
    """
    typical_values = {
        'heart': {
            'Age': '55',
            'Gender': 'Male',
            'Chest Pain Type': '1',
            'Resting Blood Pressure': '120',
            'Serum Cholesterol (mg/dl)': '200',
            'Fasting Blood Sugar > 120 mg/dl': '0',
            'Resting Electrocardiographic Results': '0',
            'Maximum Heart Rate Achieved': '150',
            'Exercise Induced Angina': '0',
            'ST Depression Induced by Exercise': '0',
            'Slope of the Peak Exercise ST Segment': '1',
            'Number of Major Vessels Colored by Fluoroscopy': '0',
            'Thalassemia': '2'
        },
        'stroke': {
            'Gender': 'Male',
            'Age': '65',
            'Hypertension': '0',
            'Heart Disease': '0',
            'Ever Married': 'Yes',
            'Work Type': '0',
            'Residence Type': '1',
            'Average Glucose Level': '100',
            'Body Mass Index': '25',
            'Smoking Status': '0'
        },
        'liver': {
            'Age': '45',
            'Gender': 'Male',
            'Total Bilirubin': '0.7',
            'Direct Bilirubin': '0.1',
            'Alkaline Phosphotase': '70',
            'Alanine Aminotransferase': '25',
            'Aspartate Aminotransferase': '25',
            'Total Proteins': '7',
            'Albumin': '4',
            'Albumin and Globulin Ratio': '1.5'
        },
        'kidney': {
            'Age': '50',
            'Blood Pressure': '120',
            'Specific Gravity': '1.015',
            'Albumin': '0',
            'Sugar': '0',
            'Red Blood Cells': 'normal',
            'Pus Cell': 'normal',
            'Pus Cell Clumps': 'notpresent',
            'Bacteria': 'notpresent',
            'Blood Glucose Random': '120',
            'Blood Urea': '30',
            'Serum Creatinine': '1.0',
            'Sodium': '135',
            'Potassium': '4.0',
            'Hemoglobin': '14',
            'Packed Cell Volume': '45',
            'White Blood Cell Count': '7500',
            'Red Blood Cell Count': '5.0',
            'Hypertension': 'no',
            'Diabetes Mellitus': 'no',
            'Coronary Artery Disease': 'no',
            'Appetite': 'good',
            'Pedal Edema': 'no',
            'Anemia': 'no'
        },
        'diabetes': {
            'Pregnancies': '2',
            'Glucose Level': '120',
            'Blood Pressure': '70',
            'Skin Thickness': '20',
            'Insulin Level': '80',
            'Body Mass Index': '25',
            'Diabetes Pedigree Function': '0.5',
            'Age': '45'
        },
        'parkinsons': {
            'Average Vocal Fundamental Frequency': '120',
            'Maximum Vocal Fundamental Frequency': '150',
            'Minimum Vocal Fundamental Frequency': '100',
            'Jitter (%)': '0.005',
            'Jitter (Abs)': '0.00004',
            'Relative Average Perturbation': '0.003',
            'Pitch Period Perturbation Quotient': '0.003',
            'DDP': '0.009',
            'Shimmer': '0.03',
            'Shimmer (dB)': '0.3',
            'APQ3': '0.015',
            'APQ5': '0.02',
            'APQ': '0.025',
            'DDA': '0.045',
            'Noise-to-Harmonics Ratio': '0.015',
            'Harmonics-to-Noise Ratio': '20',
            'Recurrence Period Density Entropy': '0.5',
            'Detrended Fluctuation Analysis': '0.7',
            'Spread 1': '-5',
            'Spread 2': '0.2',
            'D2': '2.5',
            'Pitch Period Entropy': '0.2'
        }
    }
    
    return typical_values.get(disease_type, {})

def detect_disease_type_from_fields(extracted_data):
    """
    Detect disease type based on fields in the extracted data.
    Returns the most likely disease type.
    """
    # Check for specific disease indicators
    disease_indicators = {
        'heart': ['Chest Pain Type', 'Thalassemia', 'Maximum Heart Rate Achieved'],
        'diabetes': ['Pregnancies', 'Glucose Level', 'Insulin Level', 'Diabetes Pedigree Function'],
        'stroke': ['Ever Married', 'Work Type', 'Residence Type', 'Smoking Status'],
        'liver': ['Direct Bilirubin', 'Total Bilirubin', 'Albumin and Globulin Ratio'],
        'kidney': ['Specific Gravity', 'Pus Cell', 'Pus Cell Clumps', 'Bacteria', 'Pedal Edema'],
        'parkinsons': ['Jitter (%)', 'Shimmer', 'Noise-to-Harmonics Ratio', 'DDP', 'Pitch Period Entropy']
    }
    
    # Count matches for each disease
    disease_matches = {disease: 0 for disease in disease_indicators}
    
    for disease, indicators in disease_indicators.items():
        for indicator in indicators:
            if indicator in extracted_data:
                disease_matches[disease] += 1
    
    # Find disease with most matches
    max_matches = 0
    detected_disease = None
    
    for disease, matches in disease_matches.items():
        if matches > max_matches:
            max_matches = matches
            detected_disease = disease
    
    # Only return a detected disease if we have a reasonable number of matches
    if max_matches > 1:
        return detected_disease
    return None

def get_field_names_for_disease(disease: str) -> List[str]:
    if disease not in FEATURES_BY_DISEASE:
        raise ValueError(f"Unknown disease type: {disease}")
    return FEATURES_BY_DISEASE[disease]


def clean_value(val: str, known_fields: list) -> str:
    for field in known_fields:
        val = val.replace(field, "")
    return val.strip()


def map_categorical_values(data: Dict[str, str], disease: str) -> Dict[str, str]:
    mappings = CATEGORICAL_MAPPINGS.get(disease, {})
    
    # Handle case-insensitive field matching
    for field, field_map in mappings.items():
        # Check if the field exists (case-insensitive)
        field_lower = field.lower()
        field_exists = False
        existing_field = None
        
        for key in data:
            if key.lower() == field_lower:
                field_exists = True
                existing_field = key
                break
                
        if field_exists:
            raw_val = data[existing_field]
            if raw_val is not None:
                if isinstance(raw_val, str):
                    cleaned_val = raw_val.strip()
                    # Try to find a case-insensitive match
                    for map_key, map_value in field_map.items():
                        if cleaned_val.lower() == map_key.lower():
                            data[existing_field] = map_value
                            break
    return data


def standardize_field_values(data: Dict[str, str], disease_type: str) -> Dict[str, str]:
    """
    Standardize field values based on disease type.
    Converts string values to appropriate types and formats based on the disease.
    
    Args:
        data: Dictionary of field values
        disease_type: Type of disease ('heart', 'diabetes', etc.)
        
    Returns:
        Dictionary with standardized values
    """
    standardized = {}
    
    # Create a copy to avoid modifying the original
    if not data:
        return {}
    
    # Build a lookup of normalized field names for case-insensitive lookup
    numeric_fields_normalized = {}
    if disease_type in numeric_fields_by_disease:
        for field in numeric_fields_by_disease[disease_type]:
            numeric_fields_normalized[field.lower()] = field
        
    try:
        # Copy data to standardized dict
        for key, value in data.items():
            # Handle different value types
            if value is None:
                standardized[key] = value
                continue
                
            if isinstance(value, (int, float)):
                # Preserve numeric values as is
                standardized[key] = value
                continue
                
            if isinstance(value, str):
                value = value.replace("\n", " ").strip()
                
                # Try to convert numeric values - check in a case-insensitive way
                key_lower = key.lower()
                is_numeric_field = False
                
                # Check if field name is in numeric fields (case-insensitive)
                for num_field_lower in numeric_fields_normalized:
                    if key_lower == num_field_lower:
                        is_numeric_field = True
                        break
                
                if is_numeric_field:
                    try:
                        # Extract numeric part if there's text mixed in
                        numeric_match = re.search(r'([\d\.]+)', value)
                        if numeric_match:
                            numeric_val = numeric_match.group(1)
                            if '.' in numeric_val:
                                value = float(numeric_val)
                            else:
                                value = int(numeric_val)
                    except (ValueError, TypeError):
                        # If conversion fails, keep as string
                        pass
                        
            standardized[key] = value
            
        # Apply categorical mappings if available (case-insensitive)
        if disease_type in CATEGORICAL_MAPPINGS:
            for field, mappings in CATEGORICAL_MAPPINGS.get(disease_type, {}).items():
                field_lower = field.lower()
                
                # Find field in standardized data (case-insensitive)
                for std_field in list(standardized.keys()):
                    if std_field.lower() == field_lower:
                        field_value = standardized[std_field]
                        # Try to match string value to categorical mapping
                        if isinstance(field_value, str):
                            # Look for closest match in mapping keys
                            field_value_lower = field_value.lower().strip()
                            for map_key, map_value in mappings.items():
                                if field_value_lower == map_key.lower():
                                    standardized[std_field] = map_value
                                    break
                        break
        
        return standardized
        
    except Exception as e:
        logging.error(f"Error standardizing field values: {str(e)}")
        return data


def extract_critical_fields(text: str, disease_type: str) -> Dict[str, str]:
    """
    Extract critical fields for specific diseases using specialized patterns.
    
    Args:
        text: Full text extracted from PDF
        disease_type: Type of disease to extract data for
        
    Returns:
        Dictionary of extracted critical fields
    """
    if not text:
        return {}
        
    critical_fields = {}
    
    # Define disease-specific extraction patterns
    if disease_type == 'stroke':
        # Extract age
        age_patterns = [
            r'age[:\s]*(\d+\.?\d*)',
            r'patient.*?(\d+)[\s]*years',
            r'(\d+)[\s]*years old',
            r'(\d+)[\s]*y\.?o',
            r'Age[:\s]*(\d+\.?\d*)'  # Added capitalized pattern
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text.lower())
            if match:
                critical_fields['Age'] = match.group(1)
                break
                
        # Extract BMI
        bmi_patterns = [
            r'bmi[:\s]*(\d+\.?\d*)',
            r'body mass index[:\s]*(\d+\.?\d*)',
            r'body mass index.*?(\d+\.?\d*)\s*kg\/m2',
            r'Body Mass Index[:\s]*(\d+\.?\d*)'  # Added capitalized pattern
        ]
        
        for pattern in bmi_patterns:
            match = re.search(pattern, text.lower())
            if match:
                critical_fields['Body Mass Index'] = match.group(1)
                break
                
        # Extract glucose level
        glucose_patterns = [
            r'glucose[:\s]*(\d+\.?\d*)',
            r'blood glucose[:\s]*(\d+\.?\d*)',
            r'average glucose level[:\s]*(\d+\.?\d*)',
            r'avg glucose[:\s]*(\d+\.?\d*)',
            r'Average Glucose Level[:\s]*(\d+\.?\d*)'  # Added capitalized pattern
        ]
        
        for pattern in glucose_patterns:
            match = re.search(pattern, text.lower())
            if match:
                critical_fields['Average Glucose Level'] = match.group(1)
                break
        
        # Extract Gender - Improved to handle text values like "Male" or "Female"
        gender_patterns = [
            r'gender[:\s]*(male|female|m|f)',
            r'Gender[:\s]*(Male|Female|M|F)',
            r'gender[:\s]*(\d+)',
            r'Gender[:\s]*(\d+)'
        ]
        
        for pattern in gender_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                gender_value = match.group(1).lower()
                # Map text values to numeric values expected by the model
                if gender_value in ["male", "m"]:
                    critical_fields['Gender'] = "1"
                elif gender_value in ["female", "f"]:
                    critical_fields['Gender'] = "0"
                else:
                    # If it's already numeric, use it as is
                    critical_fields['Gender'] = gender_value
                break
                
        # Extract Ever Married status
        ever_married_patterns = [
            r'ever married[:\s]*(yes|no|y|n)',
            r'marital status[:\s]*(married|single|yes|no)',
            r'married[:\s]*(yes|no|y|n)',
            r'Ever Married[:\s]*(Yes|No|Y|N)'  # Added capitalized pattern
        ]
        
        for pattern in ever_married_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).lower()
                if value in ["yes", "y", "married"]:
                    critical_fields['Ever Married'] = "Yes"
                elif value in ["no", "n", "single"]:
                    critical_fields['Ever Married'] = "No"
                else:
                    critical_fields['Ever Married'] = value
                break
        
        # Extract Residence Type - Improved to handle text values like "Urban" or "Rural"
        residence_patterns = [
            r'residence type[:\s]*(urban|rural|u|r)',
            r'residence[:\s]*(urban|rural|u|r)',
            r'Residence Type[:\s]*(Urban|Rural|U|R)',  # Added capitalized pattern
            r'residence type[:\s]*(\d+)',
            r'Residence Type[:\s]*(\d+)'
        ]
        
        for pattern in residence_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                residence_value = match.group(1).lower()
                # Map text values to numeric values expected by the model
                if residence_value in ["urban", "u"]:
                    critical_fields['Residence Type'] = "1"
                elif residence_value in ["rural", "r"]:
                    critical_fields['Residence Type'] = "0"
                else:
                    # If it's already numeric, use it as is
                    critical_fields['Residence Type'] = residence_value
                break
        
        # Extract Work Type
        work_type_patterns = [
            r'work type[:\s]*(private|self-employed|govt_job|children|never_worked|\d+)',
            r'occupation[:\s]*(private|self-employed|govt_job|children|never_worked|\d+)',
            r'employment[:\s]*(private|self-employed|govt_job|children|never_worked|\d+)',
            r'Work Type[:\s]*(Private|Self-employed|Govt_job|Children|Never_worked|\d+)'  # Added capitalized pattern
        ]
        
        for pattern in work_type_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                work_value = match.group(1).lower()
                # Map text values to numeric values expected by the model
                if work_value == "private":
                    critical_fields['Work Type'] = "0"
                elif work_value == "self-employed":
                    critical_fields['Work Type'] = "1"
                elif work_value == "govt_job":
                    critical_fields['Work Type'] = "2"
                elif work_value == "children":
                    critical_fields['Work Type'] = "3"
                elif work_value == "never_worked":
                    critical_fields['Work Type'] = "4"
                else:
                    # If it's already numeric, use it as is
                    critical_fields['Work Type'] = work_value
                break
                
        # Extract Smoking Status
        smoking_patterns = [
            r'smoking status[:\s]*(never smoked|formerly smoked|smokes|unknown|\d+)',
            r'smoke[:\s]*(never smoked|formerly smoked|smokes|unknown|\d+)',
            r'smoking[:\s]*(never smoked|formerly smoked|smokes|unknown|\d+)',
            r'Smoking Status[:\s]*(Never smoked|Formerly smoked|Smokes|Unknown|\d+)'  # Added capitalized pattern
        ]
        
        for pattern in smoking_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                smoke_value = match.group(1).lower()
                # Map text values to numeric values expected by the model
                if smoke_value == "never smoked":
                    critical_fields['Smoking Status'] = "0"
                elif smoke_value == "formerly smoked":
                    critical_fields['Smoking Status'] = "1"
                elif smoke_value == "smokes":
                    critical_fields['Smoking Status'] = "2"
                elif smoke_value == "unknown":
                    critical_fields['Smoking Status'] = "3"
                else:
                    # If it's already numeric, use it as is
                    critical_fields['Smoking Status'] = smoke_value
                break
    
    elif disease_type == 'diabetes':
        # Extract glucose level
        glucose_patterns = [
            r'glucose[:\s]*(\d+\.?\d*)',
            r'blood glucose[:\s]*(\d+\.?\d*)',
            r'fasting(?:\s*blood)?\s*glucose[:\s]*(\d+\.?\d*)',
            r'plasma glucose[:\s]*(\d+\.?\d*)',
            r'glucose level[:\s]*(\d+\.?\d*)',
            r'Glucose Level[:\s]*(\d+\.?\d*)'  # Added capitalized pattern
        ]
        
        for pattern in glucose_patterns:
            match = re.search(pattern, text.lower())
            if match:
                critical_fields['Glucose Level'] = match.group(1)
                break
        
        # Extract BMI
        bmi_patterns = [
            r'bmi[:\s]*(\d+\.?\d*)',
            r'body mass index[:\s]*(\d+\.?\d*)',
            r'body mass index.*?(\d+\.?\d*)\s*kg\/m2',
            r'Body Mass Index[:\s]*(\d+\.?\d*)'  # Added capitalized pattern
        ]
        
        for pattern in bmi_patterns:
            match = re.search(pattern, text.lower())
            if match:
                critical_fields['Body Mass Index'] = match.group(1)
                break
        
        # Extract age
        age_patterns = [
            r'age[:\s]*(\d+)',
            r'patient.*?(\d+)[\s]*years',
            r'(\d+)[\s]*years old',
            r'(\d+)[\s]*y\.?o',
            r'Age[:\s]*(\d+)'  # Added capitalized pattern
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text.lower())
            if match:
                critical_fields['Age'] = match.group(1)
                break
                
    elif disease_type == 'liver':
        # Extract bilirubin values
        total_bilirubin_patterns = [
            r'total bilirubin[:\s]*(\d+\.?\d*)',
            r'bilirubin total[:\s]*(\d+\.?\d*)',
            r'bilirubin[,\s]total[:\s]*(\d+\.?\d*)',
            r'total\s+bili[:\s]*(\d+\.?\d*)',
            r'Total Bilirubin[:\s]*(\d+\.?\d*)'  # Added capitalized pattern
        ]
        
        for pattern in total_bilirubin_patterns:
            match = re.search(pattern, text.lower())
            if match:
                critical_fields['Total Bilirubin'] = match.group(1)
                break
                
        direct_bilirubin_patterns = [
            r'direct bilirubin[:\s]*(\d+\.?\d*)',
            r'bilirubin direct[:\s]*(\d+\.?\d*)',
            r'bilirubin[,\s]direct[:\s]*(\d+\.?\d*)',
            r'direct\s+bili[:\s]*(\d+\.?\d*)',
            r'conjugated bilirubin[:\s]*(\d+\.?\d*)',
            r'Direct Bilirubin[:\s]*(\d+\.?\d*)'  # Added capitalized pattern
        ]
        
        for pattern in direct_bilirubin_patterns:
            match = re.search(pattern, text.lower())
            if match:
                critical_fields['Direct Bilirubin'] = match.group(1)
                break
                
        # Extract Gender - Improved to handle text values
        gender_patterns = [
            r'gender[:\s]*(male|female|m|f)',
            r'Gender[:\s]*(Male|Female|M|F)',
            r'gender[:\s]*(\d+)',
            r'Gender[:\s]*(\d+)'
        ]
        
        for pattern in gender_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                gender_value = match.group(1).lower()
                # Map text values to numeric values expected by the model
                if gender_value in ["male", "m"]:
                    critical_fields['Gender'] = "1"
                elif gender_value in ["female", "f"]:
                    critical_fields['Gender'] = "0"
                else:
                    # If it's already numeric, use it as is
                    critical_fields['Gender'] = gender_value
                break
                
        # Extract alkaline phosphatase
        alkaline_patterns = [
            r'alkaline phosphotase[:\s]*(\d+\.?\d*)',
            r'alkaline phosphatase[:\s]*(\d+\.?\d*)',
            r'alk\.?\s*phos[:\s]*(\d+\.?\d*)',
            r'alp[:\s]*(\d+\.?\d*)',
            r'Alkaline Phosphotase[:\s]*(\d+\.?\d*)'  # Added capitalized pattern
        ]
        
        for pattern in alkaline_patterns:
            match = re.search(pattern, text.lower())
            if match:
                critical_fields['Alkaline Phosphotase'] = match.group(1)
                break
                
        # Extract ALT (Alanine Aminotransferase)
        alt_patterns = [
            r'al[a|e]nine\s*aminotransferase[:\s]*([\d\.]+)',
            r'(?:al[a|e]nine|alt)[:\s]*([\d\.]+)(?:\s*IU\/L|\s*U\/L)?',
            r'alt(?:\s*\(sgpt\))?[:\s]*([\d\.]+)',
            r'sgpt[:\s]*([\d\.]+)',
            r'alt.*?(\d+)(?:\s*IU\/L|\s*U\/L|\s*U)?',
            r'Alanine Aminotransferase[:\s]*([\d\.]+)'  # Added capitalized pattern
        ]
        
        for pattern in alt_patterns:
            match = re.search(pattern, text.lower())
            if match:
                critical_fields['Alanine Aminotransferase'] = match.group(1)
                break
                
        # Extract AST (Aspartate Aminotransferase)
        ast_patterns = [
            r'aspartate\s*aminotransferase[:\s]*([\d\.]+)',
            r'(?:aspartate|ast)[:\s]*([\d\.]+)(?:\s*IU\/L|\s*U\/L)?',
            r'ast(?:\s*\(sgot\))?[:\s]*([\d\.]+)',
            r'sgot[:\s]*([\d\.]+)',
            r'ast.*?(\d+)(?:\s*IU\/L|\s*U\/L|\s*U)?',
            r'Aspartate Aminotransferase[:\s]*([\d\.]+)'  # Added capitalized pattern
        ]
        
        for pattern in ast_patterns:
            match = re.search(pattern, text.lower())
            if match:
                critical_fields['Aspartate Aminotransferase'] = match.group(1)
                break
                
    elif disease_type == 'kidney':
        # Extract specific gravity
        sg_patterns = [
            r'specific gravity[:\s]*(\d+\.?\d*)',
            r'urine specific gravity[:\s]*(\d+\.?\d*)',
            r'sg[:\s]*(\d+\.?\d*)',
            r'Specific Gravity[:\s]*(\d+\.?\d*)'  # Added capitalized pattern
        ]
        
        for pattern in sg_patterns:
            match = re.search(pattern, text.lower())
            if match:
                critical_fields['Specific Gravity'] = match.group(1)
                break
                
        # Extract albumin
        albumin_patterns = [
            r'albumin[:\s]*(\d+\.?\d*)',
            r'serum albumin[:\s]*(\d+\.?\d*)',
            r'alb[:\s]*(\d+\.?\d*)',
            r'Albumin[:\s]*(\d+\.?\d*)'  # Added capitalized pattern
        ]
        
        for pattern in albumin_patterns:
            match = re.search(pattern, text.lower())
            if match:
                critical_fields['Albumin'] = match.group(1)
                break
                
        # Enhanced patterns for extracting kidney-specific categorical fields
        categorical_fields = [
            'Red Blood Cells', 'Pus Cell', 'Pus Cell Clumps', 'Bacteria', 
            'Hypertension', 'Diabetes Mellitus', 'Coronary Artery Disease', 
            'Appetite', 'Pedal Edema', 'Anemia'
        ]
        
        for field in categorical_fields:
            field_patterns = [
                rf'{field.lower()}[:\s]*(normal|abnormal|present|notpresent|yes|no|good|poor|[\w\s]+)',
                rf'{field}[:\s]*(Normal|Abnormal|Present|Notpresent|Yes|No|Good|Poor|[\w\s]+)'  # Capitalized pattern
            ]
            
            for pattern in field_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip().lower()
                    
                    # Map to expected values for each field
                    if field in ['Red Blood Cells', 'Pus Cell']:
                        if value in ['normal']:
                            critical_fields[field] = "0"  # normal
                        elif value in ['abnormal']:
                            critical_fields[field] = "1"  # abnormal
                        else:
                            critical_fields[field] = value
                    elif field in ['Pus Cell Clumps', 'Bacteria']:
                        if value in ['present']:
                            critical_fields[field] = "1"  # present
                        elif value in ['notpresent', 'not present']:
                            critical_fields[field] = "0"  # not present
                        else:
                            critical_fields[field] = value
                    elif field in ['Hypertension', 'Diabetes Mellitus', 'Coronary Artery Disease', 'Pedal Edema', 'Anemia']:
                        if value in ['yes', 'y']:
                            critical_fields[field] = "1"  # yes
                        elif value in ['no', 'n']:
                            critical_fields[field] = "0"  # no
                        else:
                            critical_fields[field] = value
                    elif field == 'Appetite':
                        if value in ['good', 'g']:
                            critical_fields[field] = "1"  # good
                        elif value in ['poor', 'p']:
                            critical_fields[field] = "0"  # poor
                        else:
                            critical_fields[field] = value
                    break
    
    elif disease_type == 'parkinsons':
        # Extract jitter
        jitter_patterns = [
            r'jitter[:\s]*(\d+\.?\d*)',
            r'jitter \(%\)[:\s]*(\d+\.?\d*)',
            r'jitter percent[:\s]*(\d+\.?\d*)',
            r'jitter.*?(\d+\.?\d*)(?:\s*%)?',
            r'Jitter \(%\)[:\s]*(\d+\.?\d*)'  # Added capitalized pattern
        ]
        
        for pattern in jitter_patterns:
            match = re.search(pattern, text.lower())
            if match:
                critical_fields['Jitter (%)'] = match.group(1)
                break
        
        # Extract jitter absolute
        jitter_abs_patterns = [
            r'jitter \(abs\)[:\s]*([\d\.e\-]+)',
            r'jitter absolute[:\s]*([\d\.e\-]+)',
            r'jitter \(absolute\)[:\s]*([\d\.e\-]+)',
            r'Jitter \(Abs\)[:\s]*([\d\.e\-]+)'  # Added capitalized pattern
        ]
        
        for pattern in jitter_abs_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                critical_fields['Jitter (Abs)'] = match.group(1)
                break
                
        # Extract shimmer
        shimmer_patterns = [
            r'shimmer[:\s]*(\d+\.?\d*)',
            r'shimmer.*?in decibels[:\s]*(\d+\.?\d*)',
            r'shimmer.*?(\d+\.?\d*)(?:\s*dB)?',
            r'Shimmer[:\s]*(\d+\.?\d*)'  # Added capitalized pattern
        ]
        
        for pattern in shimmer_patterns:
            match = re.search(pattern, text.lower())
            if match:
                critical_fields['Shimmer'] = match.group(1)
                break
                
        # Improved pattern for shimmer in decibels
        shimmer_db_patterns = [
            r'shimmer \(db\)[:\s]*(\d+\.?\d*)',
            r'shimmer in decibels[:\s]*(\d+\.?\d*)',
            r'shimmer db[:\s]*(\d+\.?\d*)',
            r'shimmer.*?(\d+\.?\d*)\s*dB',
            r'Shimmer \(dB\)[:\s]*(\d+\.?\d*)'  # Added capitalized pattern
        ]
        
        for pattern in shimmer_db_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                critical_fields['Shimmer (dB)'] = match.group(1)
                break
    
    elif disease_type == 'heart':
        # Extract blood pressure
        bp_patterns = [
            r'blood pressure[:\s]*(\d+)',
            r'bp[:\s]*(\d+)',
            r'resting.*?blood pressure[:\s]*(\d+)',
            r'Resting Blood Pressure[:\s]*(\d+)'  # Added capitalized pattern
        ]
        
        for pattern in bp_patterns:
            match = re.search(pattern, text.lower())
            if match:
                critical_fields['Resting Blood Pressure'] = match.group(1)
                break
                
        # Improved pattern for Serum Cholesterol
        chol_patterns = [
            r'cholesterol[:\s]*(\d+\.?\d*)',
            r'serum cholesterol[:\s]*(\d+\.?\d*)',
            r'total cholesterol[:\s]*(\d+\.?\d*)',
            r'Serum Cholesterol \(mg\/dl\)[:\s]*(\d+\.?\d*)',
            r'Serum Cholesterol[:\s]*(\d+\.?\d*)'
        ]
        
        for pattern in chol_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                critical_fields['Serum Cholesterol (mg/dl)'] = match.group(1)
                break
        
        # Extract Gender - Improved to handle text values
        gender_patterns = [
            r'gender[:\s]*(male|female|m|f)',
            r'Gender[:\s]*(Male|Female|M|F)',
            r'gender[:\s]*(\d+)',
            r'Gender[:\s]*(\d+)'
        ]
        
        for pattern in gender_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                gender_value = match.group(1).lower()
                # Map text values to numeric values expected by the model
                if gender_value in ["male", "m"]:
                    critical_fields['Gender'] = "1"
                elif gender_value in ["female", "f"]:
                    critical_fields['Gender'] = "0"
                else:
                    # If it's already numeric, use it as is
                    critical_fields['Gender'] = gender_value
                break
                
        # Extract Number of Major Vessels Colored by Fluoroscopy
        vessels_patterns = [
            r'major vessels[:\s]*(\d+)',
            r'vessels colored[:\s]*(\d+)',
            r'colored vessels[:\s]*(\d+)',
            r'fluoroscopy.*?vessels[:\s]*(\d+)',
            r'vessels.*?fluoroscopy[:\s]*(\d+)',
            r'number of major vessels[:\s]*(\d+)',
            r'Number of Major Vessels Colored by Fluoroscopy[:\s]*(\d+)'  # Added capitalized pattern
        ]
        
        for pattern in vessels_patterns:
            match = re.search(pattern, text.lower())
            if match:
                critical_fields['Number of Major Vessels Colored by Fluoroscopy'] = match.group(1)
                break
    
    # Try to find numerical values based on keywords for any other fields
    required_fields = get_field_names_for_disease(disease_type)
    for field in required_fields:
        if field not in critical_fields:
            # Create field pattern from field name (handles spaces, dashes, etc.)
            field_pattern = field.lower().replace(' ', '[\\s_-]*')
            patterns = [
                rf'{field_pattern}[:\s]*(\d+\.?\d*)',
                rf'{field_pattern}.*?(\d+\.?\d*)[\s]*(?:mg\/dl|mmol\/l|g\/dl|u\/l|iu\/l)?',
                rf'{field}[:\s]*(\d+\.?\d*)',
                rf'{field}.*?(\d+\.?\d*)[\s]*(?:mg\/dl|mmol\/l|g\/dl|u\/l|iu\/l)?'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    critical_fields[field] = match.group(1)
                    break
    
    # Check directly for field: value pattern for any fields that might have been missed
    lines = text.split('\n')
    for line in lines:
        match = KEY_VALUE_REGEX.match(line)
        if match:
            field, value = match.groups()
            field = field.strip()
            value = value.strip()
            
            # Check if this is a field we care about
            if field in required_fields and field not in critical_fields:
                critical_fields[field] = value
    
    # Log how many critical fields were extracted
    logging.info(f"Extracted {len(critical_fields)} critical fields for {disease_type}")
    return critical_fields

def extract_medical_data(pdf_path: str, disease_type: str) -> Dict[str, str]:
    """
    Extract medical data from a PDF file based on the disease type.
    
    Args:
        pdf_path: Path to the PDF file or a file-like object
        disease_type: Type of disease to extract data for (e.g., 'heart', 'diabetes'), or None to extract all fields
    
    Returns:
        Dictionary of extracted medical data
    """
    extracted_data = {}
    
    try:
        # Check if pdf_path is a string path or a file-like object
        if isinstance(pdf_path, str):
            # It's a file path
            pdf = pdfplumber.open(pdf_path)
        else:
            # It's a file-like object (e.g., from st.file_uploader)
            pdf_path.seek(0)  # Reset file pointer
            pdf = pdfplumber.open(pdf_path)
        
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        # If disease_type is None, extract all possible medical fields
        if disease_type is None:
            # Try to extract fields for all disease types
            all_extracted_data = {}
            for disease in FEATURES_BY_DISEASE.keys():
                disease_fields = extract_critical_fields(text, disease)
                # Add to the combined dictionary
                all_extracted_data.update(disease_fields)
            
            # Return the combined data from all disease types
            return all_extracted_data
            
        # If specific disease type is provided, extract only those fields
        extracted_data = extract_critical_fields(text, disease_type)
        
        # Attempt to detect fields for the specific disease type
        predicted_disease = detect_disease_type_from_fields(extracted_data)
        if predicted_disease and predicted_disease != disease_type:
            logging.info(f"PDF seems to contain data for {predicted_disease}, but requested {disease_type}")
        
        pdf.close()
        return extracted_data
    
    except Exception as e:
        logging.error(f"Error extracting data from PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return extracted_data


def validate_extracted_data(extracted_data, selected_disease):
    """
    Validate the extracted data to ensure all required fields for the disease are present.
    Returns validated data and a list of missing fields.
    """
    try:
        # Get required fields for the disease
        required_fields = get_field_names_for_disease(selected_disease)
        
        # Normalize the keys for case-insensitive comparison
        normalized_data = {}
        field_mapping = {}  # Maps normalized names to original names
        
        for key, value in extracted_data.items():
            normalized_key = key.lower()
            normalized_data[normalized_key] = value
            field_mapping[normalized_key] = key
        
        # Check for missing fields (case-insensitive)
        missing_fields = []
        for required_field in required_fields:
            # Skip optional fields in validation
            if required_field in OPTIONAL_FIELDS:
                continue
                
            normalized_required = required_field.lower()
            
            # Check if field exists in a case-insensitive way
            if normalized_required in normalized_data:
                value = normalized_data[normalized_required]
                # Check if value is truly empty (None or empty string, but not zero)
                is_empty = (value is None or 
                           (isinstance(value, str) and value.strip() == ""))
                if is_empty:
                    missing_fields.append(required_field)
            else:
                missing_fields.append(required_field)
        
        # Add typical values for missing fields if any
        if missing_fields:
            logging.warning(f"Missing critical fields for {selected_disease}: {missing_fields}")
            # Use a helper function to get typical values
            typical_values = get_typical_values(selected_disease)
            
            for field in missing_fields:
                if field in typical_values:
                    # Add to the original extracted_data
                    extracted_data[field] = typical_values[field]
                    # Also update normalized data
                    normalized_data[field.lower()] = typical_values[field]
                    field_mapping[field.lower()] = field
                    logging.info(f"Added typical value for {field}: {typical_values[field]}")
        
        # Create validated data with proper field names from field_mapping
        validated_data = {}
        for norm_key, value in normalized_data.items():
            original_key = field_mapping.get(norm_key, norm_key)
            validated_data[original_key] = value
        
        # Process Parkinson's fields with safe_cast_float for numeric values
        if selected_disease == 'parkinsons':
            for field in required_fields:
                field_lower = field.lower()
                
                # Find the field in validated_data (case-insensitive)
                for val_field in list(validated_data.keys()):
                    if val_field.lower() == field_lower:
                        # Apply safe float casting to numeric fields
                        if field in numeric_fields_by_disease.get(selected_disease, []):
                            current_val = validated_data[val_field]
                            float_val = safe_cast_float(current_val)
                            if float_val is not None:
                                validated_data[val_field] = float_val
                            elif current_val is not None:
                                logging.warning(f"Could not convert Parkinson's field {field} value '{current_val}' to float")
                        break
        
        # Process default categorical fields and ensure proper integer casting
        if selected_disease in INTEGER_FIELDS:
            for int_field in INTEGER_FIELDS[selected_disease]:
                int_field_lower = int_field.lower()
                
                # Find the field in validated_data (case-insensitive)
                found = False
                for val_field in list(validated_data.keys()):
                    if val_field.lower() == int_field_lower:
                        found = True
                        # Try to convert to integer
                        try:
                            current_val = validated_data[val_field]
                            if current_val is not None:
                                if isinstance(current_val, str):
                                    if current_val.strip() == "":
                                        validated_data[val_field] = 0
                                    else:
                                        # Try to extract numeric part
                                        match = re.search(r'(\d+)', current_val)
                                        if match:
                                            validated_data[val_field] = int(match.group(1))
                                        else:
                                            # Maps common yes/no values
                                            if current_val.lower() in ['yes', 'y', 'true', 't']:
                                                validated_data[val_field] = 1
                                            elif current_val.lower() in ['no', 'n', 'false', 'f']:
                                                validated_data[val_field] = 0
                                elif isinstance(current_val, (int, float)):
                                    validated_data[val_field] = int(current_val)
                        except (ValueError, TypeError):
                            logging.warning(f"Could not convert {int_field} value '{validated_data[val_field]}' to integer")
                        break
                        
                # Add default value of 0 if field not found
                if not found and int_field not in missing_fields:
                    validated_data[int_field] = 0
                    logging.info(f"Added default integer 0 for field {int_field}")
        
        # Standardize the validated data
        validated_data = standardize_field_values(validated_data, selected_disease)
        
        # Check if standardization was successful
        standardization_issues = []
        for field in required_fields:
            # Skip optional fields in standardization check
            if field in OPTIONAL_FIELDS:
                continue
                
            # Check if field is supposed to be numeric after standardization
            field_lower = field.lower()
            field_found = False
            
            for validated_field in validated_data:
                if validated_field.lower() == field_lower:
                    field_found = True
                    if (field in numeric_fields_by_disease.get(selected_disease, []) and 
                        not isinstance(validated_data[validated_field], (int, float))):
                        standardization_issues.append(field)
                    break
            
            if not field_found and field in numeric_fields_by_disease.get(selected_disease, []):
                standardization_issues.append(field)
        
        # Determine if detected disease matches selected disease
        detected_disease = detect_disease_type_from_fields(extracted_data)
        disease_mismatch = detected_disease and detected_disease != selected_disease
        
        # Return in a format compatible with dashboard.py
        return {
            "validated_data": validated_data,
            "missing_fields": missing_fields,
            "standardization_issues": standardization_issues,
            "is_valid": len(missing_fields) == 0 and len(standardization_issues) == 0,
            "detected_disease": detected_disease,
            "disease_mismatch": disease_mismatch
        }
    except Exception as e:
        logging.error(f"Error validating extracted data: {str(e)}")
        logging.exception("Detailed error:")
        return {
            "validated_data": {},
            "missing_fields": required_fields,
            "standardization_issues": [],
            "is_valid": False,
            "detected_disease": None,
            "disease_mismatch": False
        }
