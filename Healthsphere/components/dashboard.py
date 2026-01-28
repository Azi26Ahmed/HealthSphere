import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import os
import json
import tempfile
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from components.database import (register_user, verify_user, update_username, 
                               update_password, get_user_predictions, 
                               get_user_prediction_parameters, get_user_chat_history,
                               save_chat_history, delete_chat_history, db,
                               get_prediction_history_by_date, get_disease_specific_stats,
                               get_parameter_trends, get_prediction_stats, delete_prediction)
from components.prediction import DiseasePredictor
from components.visualization import Visualizer
from components.settings import show_settings
from components.utils import load_lottie_file, safe_parse_datetime
import PyPDF2
from components.pdf_extraction import get_field_names_for_disease
from components.statistics import show_stats

# Initialize predictor and visualizer at module level
predictor = DiseasePredictor()
visualizer = Visualizer()

# Get animation file path
animation_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Animation", "animation.json")
animation = load_lottie_file(animation_path)

def save_prediction(username, disease_type, input_data, prediction_result, probability=None):
    """Save prediction to database"""
    try:
        timestamp = datetime.now()
        prediction_doc = {
            'username': username,
            'disease_type': disease_type,
            'input_data': input_data,
            'prediction_result': bool(prediction_result),
            'probability': probability,  # Keep consistent with database schema
            'timestamp': timestamp
        }
        
        # Save to collections
        collection_name = f"{disease_type.lower().replace(' ', '_')}_predictions"
        db[collection_name].insert_one(prediction_doc)
        db.predictions.insert_one(prediction_doc)
        
    except Exception as e:
        print(f"Error saving prediction: {str(e)}")

def process_pdf(predictor, disease_type, uploaded_file):
    """Process uploaded PDF file and make prediction"""
    try:
        # Hide the default file uploader info messages
        uploaded_file.seek(0)
        file_content = uploaded_file.getvalue()
        file_size = len(file_content) / 1024  # Size in KB
        
        st.info(f"Processing PDF file: {uploaded_file.name}, Size: {file_size:.2f} KB")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        # Get feature info for the selected disease
        feature_info = predictor.get_feature_info(disease_type)
        if not feature_info:
            st.error(f"No feature information found for {disease_type}")
            return
        
        # Extract and validate features from PDF
        from components.pdf_extraction import extract_medical_data, validate_extracted_data, get_field_names_for_disease, standardize_field_values
        import logging
        
        # Enable logging to see what's happening
        logging.basicConfig(level=logging.INFO)
        
        # Reset the file pointer to the beginning
        uploaded_file.seek(0)
        
        # Extract data from the uploaded file
        extracted_data = None
        try:
            st.text("Extracting data from PDF file...")
            extracted_data = extract_medical_data(uploaded_file, disease_type)  # ✅ Correct
            if extracted_data:
                st.success("Successfully extracted data from PDF")
        except Exception as e:
            st.error(f"Error in direct extraction: {str(e)}")
            logging.exception("Detailed error in direct extraction:")
            extracted_data = None
        
        # If direct extraction failed, try with the saved file path
        if not extracted_data:
            try:
                st.text("Trying alternative extraction method...")
                extracted_data = extract_medical_data(tmp_file_path, disease_type)
                validated_data = validate_extracted_data(extracted_data, disease_type)
                
                if extracted_data:
                    st.success("Successfully extracted data using alternative method")
            except Exception as e:
                st.error(f"Error in file path extraction: {str(e)}")
                logging.exception("Detailed error in file path extraction:")
                extracted_data = None
        
        if not extracted_data:
            st.error("No data could be extracted from the PDF. Make sure the PDF has text content (not just images).")
            # Show debugging information in an expander
            with st.expander("Debugging Information"):
                st.text("File metadata:")
                st.json({
                    "filename": uploaded_file.name,
                    "size_kb": file_size,
                    "temp_path": tmp_file_path
                })
                
                # Try to extract some raw text for debugging
                try:
                    uploaded_file.seek(0)
                    with open(tmp_file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        raw_text = ""
                        for i, page in enumerate(reader.pages):
                            page_text = page.extract_text()
                            raw_text += f"--- Page {i+1} ---\n{page_text}\n\n"
                        
                        if raw_text.strip():
                            st.text("Raw text extracted from PDF:")
                            st.text_area("Raw text", raw_text, height=200)
                        else:
                            st.warning("No text could be extracted from this PDF. It may contain only images or be protected.")
                except Exception as e:
                    st.error(f"Error getting raw text: {str(e)}")
            
            # Option to proceed with manual input
            st.info("Would you like to enter the values manually instead?")
            if st.button("Enter Values Manually"):
                # Create manual input form with all required fields
                required_fields = get_field_names_for_disease(disease_type)
                st.subheader("Manual Data Entry Form")
                
                # Create a form for manual data entry
                with st.form(key="manual_data_entry"):
                    manual_data = {}
                    col1, col2 = st.columns(2)
                    use_col1 = True
                    
                    for field in required_fields:
                        current_col = col1 if use_col1 else col2
                        use_col1 = not use_col1
                        
                        with current_col:
                            if field.lower() in ["gender", "sex"]:
                                value = st.selectbox(f"{field}:", ["Male", "Female"])
                            elif any(categorical_term in field.lower() for categorical_term in ["type", "result", "induced", "status"]):
                                value = st.number_input(f"{field}:", min_value=0, max_value=10, step=1)
                            else:
                                value = st.number_input(f"{field}:", step=0.1)
                            
                            manual_data[field] = value
                    
                    if st.form_submit_button("Submit Manual Values"):
                        # Use the manually entered data as our extracted data
                        extracted_data = manual_data
                        st.success("Successfully processed manual input!")
                    else:
                        return
            else:
                return
        
        # Show the actual extracted data in an expander
        with st.expander("View Raw Extracted Data"):
            st.json(extracted_data)
        
        # Get required fields for this disease
        required_fields = get_field_names_for_disease(disease_type)
        
        # Reset the file pointer to the beginning again
        uploaded_file.seek(0)
        
        # Validate the extracted data against required fields
        validation_result = validate_extracted_data(extracted_data, disease_type)
        validated_data = validation_result["validated_data"]
        missing_fields = validation_result["missing_fields"]
        detected_disease = validation_result["detected_disease"]
        
        # If validation returned None, it means we need manual input and should wait
        if validated_data is None:
            st.error("Validation failed. Could not process extracted data.")
            return
        
        # Standardize field values based on disease type
        validated_data = standardize_field_values(validated_data, disease_type)
        
        # Show the validated data
        with st.expander("View Validated Data"):
            st.json(validated_data)
        
        # Update critical fields list to match the new column names
        critical_fields_by_disease = {
            'heart': ['Age', 'Gender', 'Chest Pain Type', 'Resting Blood Pressure', 'Serum Cholesterol (mg/dl)'],
            'diabetes': ['Glucose Level', 'Body Mass Index', 'Age'],
            'liver': ['Total Bilirubin', 'Direct Bilirubin', 'Alkaline Phosphotase', 'Alanine Aminotransferase', 'Aspartate Aminotransferase'],
            'kidney': ['Blood Pressure', 'Specific Gravity', 'Albumin'],
            'stroke': ['Age', 'Body Mass Index', 'Average Glucose Level'],
            'parkinsons': ['Average Vocal Fundamental Frequency', 'Maximum Vocal Fundamental Frequency', 'Jitter (%)', 'Shimmer']
        }
        
        # Check if any critical fields are missing in the validated data
        if disease_type in critical_fields_by_disease:
            critical_missing = [field for field in critical_fields_by_disease[disease_type] if field not in validated_data]
            if critical_missing:
                st.error(f"Critical fields are missing: {', '.join(critical_missing)}")
                st.info("Please ensure your PDF contains these important parameters or enter them manually for an accurate prediction.")
                
                # Show a simplified form for just the critical missing fields
                st.write("### Enter Missing Critical Fields")
                # Create columns for form layout
                cols = st.columns(min(3, len(critical_missing)))
                manually_filled = {}
                
                for i, field in enumerate(critical_missing):
                    col_idx = i % len(cols)
                    with cols[col_idx]:
                        field_type = feature_info['types'].get(field, 'numeric')
                        field_label = feature_info['labels'].get(field, field)
                        
                        if field_type == 'categorical':
                            options = feature_info['categories'].get(field, [])
                            value = st.selectbox(f"{field_label}:", options, key=f"missing_{field}")
                        else:
                            # For numeric fields, offer a preselected typical value as a suggestion
                            typical_values = {
                                'Age': 45.0,
                                'Gender': 1,  # Male
                                'Chest Pain Type': 1, 
                                'Resting Blood Pressure': 120.0,
                                'Serum Cholesterol (mg/dl)': 200.0,
                                'Glucose Level': 120.0,
                                'Body Mass Index': 24.0,
                                'Total Bilirubin': 0.8,
                                'Direct Bilirubin': 0.2,
                                'Alkaline Phosphotase': 80.0,
                                'Alanine Aminotransferase': 25.0,
                                'Aspartate Aminotransferase': 25.0,
                                'Blood Pressure': 120.0,
                                'Specific Gravity': 1.015,
                                'Albumin': 0.0,
                                'Average Glucose Level': 110.0,
                                'Average Vocal Fundamental Frequency': 150.0,
                                'Maximum Vocal Fundamental Frequency': 200.0,
                                'Jitter (%)': 0.006,
                                'Shimmer': 0.03
                            }
                            
                            default_value = typical_values.get(field, 0.0)
                            value = st.number_input(
                                f"{field_label}:", 
                                value=default_value,
                                step=0.1 if default_value < 10 else 1.0,
                                key=f"missing_{field}"
                            )
                        
                        manually_filled[field] = value
                
                # More prominent, user-friendly button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("⚕️ Add Missing Fields and Continue", type="primary"):
                        # Update validated_data with manually filled values
                        validated_data.update(manually_filled)
                        st.success("Added missing fields! Continuing with prediction...")
                    else:
                        return
                        
        # At this point, validated_data contains the data we want to use
        input_data = validated_data
            
        # If detected disease type doesn't match selected disease type, show a warning
        if detected_disease and detected_disease != disease_type:
            st.warning(f"The uploaded PDF appears to be for {detected_disease} disease, but you selected {disease_type}.")
            proceed_anyway = st.checkbox("Proceed anyway?")
            if not proceed_anyway:
                return
            
        # If we still don't have enough data, show error
        if not input_data:
            st.error("Could not extract enough relevant data from the PDF for prediction.")
            return
        
        # Display extracted data with human-readable values
        st.write("### Review Extracted Data")
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        use_col1 = True
        
        # Show critical fields first, then others
        critical_fields = critical_fields_by_disease.get(disease_type, [])
        sorted_features = sorted(feature_info['features'], key=lambda x: (x not in critical_fields, x))
        
        for feature in sorted_features:
            if feature in input_data:
                current_col = col1 if use_col1 else col2
                use_col1 = not use_col1
                
                with current_col:
                    # Get feature type and label
                    feature_type = feature_info['types'].get(feature, 'numeric')
                    feature_label = feature_info.get('labels', {}).get(feature, feature)
                    
                    value = input_data[feature]
                    
                    # Convert numeric categorical values to human-readable text
                    if feature_type == 'categorical' and disease_type in predictor.reverse_mappings:
                        if feature in predictor.reverse_mappings[disease_type]:
                            try:
                                numeric_value = int(float(value))
                                if numeric_value in predictor.reverse_mappings[disease_type][feature]:
                                    display_value = predictor.reverse_mappings[disease_type][feature][numeric_value]
                                    st.info(f"**{feature_label}**: {display_value} ({numeric_value})")
                                    continue
                            except (ValueError, TypeError):
                                pass
                    
                    # For numeric values or if conversion failed
                    if isinstance(value, (int, float)):
                        st.info(f"**{feature_label}**: {value:.2f}")
                    else:
                        st.info(f"**{feature_label}**: {value}")
            else:
                # For missing features, show placeholder in italics
                if feature in missing_fields:
                    current_col = col1 if use_col1 else col2
                    use_col1 = not use_col1
                    with current_col:
                        feature_label = feature_info.get('labels', {}).get(feature, feature)
                        st.warning(f"**{feature_label}**: *Missing*")
        
        # Show an indicator of data completeness
        if missing_fields:
            completeness = (len(required_fields) - len(missing_fields)) / len(required_fields) * 100
            st.write(f"**Data Completeness**: {completeness:.1f}%")
            st.warning(f"Some fields will use default/estimated values: {', '.join(missing_fields)}")
        
        # Offer a button to proceed with incomplete data or cancel
        if missing_fields and not st.button("Proceed with available data"):
            st.stop()
                
        # Try to make prediction with available data
        try:
            result, error = predictor.predict(disease_type, input_data)
            
            if error:
                st.error(f"Error processing PDF: {error}")
                return
            
            # Display prediction result prominently
            st.subheader("Prediction Result")
            if result and isinstance(result, dict):  # Check that result is a dictionary
                show_prediction_result(result, disease_type)
                
                # Display confidence information
                if missing_fields:
                    st.info("Note: This prediction was made with some missing or estimated fields, which may affect accuracy.")
                
                # Save prediction to database if user is logged in
                if 'username' in st.session_state:
                    save_prediction(
                        st.session_state['username'],
                        disease_type,
                        input_data,
                        result['prediction'],
                        result.get('probability', result.get('prediction_probability', 0))
                    )
                    
                    # Display success message
                    st.success("Prediction saved to your account history!")
            else:
                st.error("Could not make prediction from the extracted data")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.exception(e)  # Show detailed error info
        
        # Clean up temporary file
        try:
            os.unlink(tmp_file_path)
        except Exception as e:
            print(f"Error removing temporary file: {str(e)}")
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        # Print detailed error for debugging
        print(f"Detailed error: {str(e)}")
        import traceback
        traceback.print_exc()

def show_manual_input_form(disease_type):
    """Show manual input form for disease prediction"""
    predictor = DiseasePredictor()
    feature_info = predictor.get_feature_info(disease_type)
    
    if not feature_info:
        st.error(f"No feature information found for {disease_type}")
        return
    
    st.write("Please enter the following information:")
    input_data = {}
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    # Alternate between columns for better layout
    use_col1 = True
    
    for feature in feature_info['features']:
        current_col = col1 if use_col1 else col2
        use_col1 = not use_col1
        
        # Get feature type and label
        feature_type = feature_info['types'].get(feature, 'numeric')
        feature_label = feature_info.get('labels', {}).get(feature, feature)
        
        with current_col:
            if feature_type == 'categorical':
                # Use select box for categorical features with human-readable options
                options = feature_info['categories'].get(feature, [])
                if options:
                    value = st.selectbox(
                        feature_label,
                        options=options,
                        help=f"Select {feature_label.lower()}"
                    )
                    # Convert categorical value to numeric for model
                    input_data[feature] = predictor.categorical_to_numeric(disease_type, feature, value)
            else:
                # Use number input for numeric features
                ranges = feature_info.get('ranges', {}).get(feature, '0-100')
                try:
                    if ranges and '-' in ranges:
                        min_val, max_val = map(float, ranges.split('-'))
                    else:
                        # Default range if not specified or invalid
                        min_val, max_val = 0, 100
                except ValueError:
                    # Fallback if conversion fails
                    st.warning(f"Invalid range format for {feature_label}: {ranges}. Using default range (0-100).")
                    min_val, max_val = 0, 100
                
                value = st.number_input(
                    feature_label,
                    min_value=min_val,
                    max_value=max_val,
                    help=f"Enter {feature_label.lower()} (range: {ranges if ranges else '0-100'})"
                )
                input_data[feature] = value
    
    if st.button("Predict"):
        result, error = predictor.predict(disease_type, input_data)
        if error:
            st.error(f"Error making prediction: {error}")
        else:
            show_prediction_result(result, disease_type)
            
            # Save prediction to database if user is logged in
            if 'username' in st.session_state:
                # Safely get probability value using .get() method
                probability = result.get('probability', result.get('prediction_probability', 0))
                save_prediction(
                    st.session_state['username'],
                    disease_type,
                    input_data,
                    result['prediction'],
                    probability
                )

def process_smartmatch(uploaded_file):
    """Process PDF file using SmartMatch to predict multiple diseases with available data"""
    try:
        # Hide the default file uploader info messages
        uploaded_file.seek(0)
        file_content = uploaded_file.getvalue()
        file_size = len(file_content) / 1024  # Size in KB
        
        st.info(f"Processing PDF file: {uploaded_file.name}, Size: {file_size:.2f} KB")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        # Extract data from the uploaded file
        from components.pdf_extraction import extract_medical_data, validate_extracted_data, get_typical_values
        import logging
        
        # Enable logging to see what's happening
        logging.basicConfig(level=logging.INFO)
        
        # Reset the file pointer to the beginning
        uploaded_file.seek(0)
        
        # Extract data without specifying a disease type to get all possible fields
        st.text("Extracting data from PDF file...")
        extracted_data = None
        
        try:
            # Extract all data from PDF without filtering for a specific disease
            extracted_data = extract_medical_data(uploaded_file, None)
            if extracted_data:
                st.success("Successfully extracted data from PDF")
        except Exception as e:
            st.error(f"Error in direct extraction: {str(e)}")
            extracted_data = None
        
        # If direct extraction failed, try with the saved file path
        if not extracted_data:
            try:
                st.text("Trying alternative extraction method...")
                extracted_data = extract_medical_data(tmp_file_path, None)
                
                if extracted_data:
                    st.success("Successfully extracted data using alternative method")
            except Exception as e:
                st.error(f"Error in file path extraction: {str(e)}")
                extracted_data = None
        
        if not extracted_data:
            st.error("No data could be extracted from the PDF. Make sure the PDF has text content (not just images).")
            return
        
        # Show the actual extracted data in an expander
        with st.expander("View Raw Extracted Data"):
            st.json(extracted_data)
        
        # Initialize predictor
        predictor = DiseasePredictor()
        
        # Map disease names for display
        disease_display_map = {
            'heart': 'Heart Disease',
            'diabetes': 'Diabetes',
            'liver': 'Liver Disease',
            'kidney': 'Kidney Disease',
            'stroke': 'Stroke',
            'parkinsons': 'Parkinson\'s Disease'
        }
        
        # For each disease, check if we have enough features to make a prediction
        st.subheader("Compatible Disease Predictions")
        
        # Dictionary to store prediction results for compatible diseases
        predictions = {}
        skipped_diseases = []
        
        # Define critical fields that must be present for each disease
        critical_fields_by_disease = {
            'heart': ['Age', 'Gender', 'Chest Pain Type', 'Resting Blood Pressure', 'Serum Cholesterol (mg/dl)'],
            'diabetes': ['Glucose Level', 'Body Mass Index', 'Age'],
            'liver': ['Total Bilirubin', 'Direct Bilirubin', 'Alkaline Phosphotase', 'Alanine Aminotransferase', 'Aspartate Aminotransferase'],
            'kidney': ['Blood Pressure', 'Specific Gravity', 'Albumin'],
            'stroke': ['Age', 'Body Mass Index', 'Average Glucose Level'],
            'parkinsons': ['Average Vocal Fundamental Frequency', 'Maximum Vocal Fundamental Frequency', 'Jitter (%)', 'Shimmer']
        }
        
        # Process each disease type
        for disease_type in ['heart', 'diabetes', 'liver', 'kidney', 'stroke', 'parkinsons']:
            # Get required features for this disease
            feature_info = predictor.get_feature_info(disease_type)
            if not feature_info:
                skipped_diseases.append(disease_display_map[disease_type])
                continue
                
            # Validate extracted data against this disease's required features
            validation_result = validate_extracted_data(extracted_data, disease_type)
            validated_data = validation_result["validated_data"]
            missing_fields = validation_result["missing_fields"]
            
            # Get typical values for this disease
            typical_values = get_typical_values(disease_type)
            
            # Check if critical fields are missing or using default values
            critical_fields = critical_fields_by_disease.get(disease_type, [])
            critical_missing = []
            critical_using_defaults = []
            
            for field in critical_fields:
                normalized_field = None
                field_found = False
                
                # Check for case-insensitive field matching
                for key in validated_data:
                    if key.lower() == field.lower():
                        normalized_field = key
                        field_found = True
                        break
                
                if not field_found:
                    critical_missing.append(field)
                elif normalized_field:
                    # Check if this field is using a default value
                    current_value = str(validated_data[normalized_field])
                    typical_value = str(typical_values.get(field, ''))
                    
                    if current_value == typical_value:
                        critical_using_defaults.append(field)
            
            # Calculate what percentage of critical fields are using defaults
            total_critical = len(critical_fields)
            if total_critical > 0:
                default_percent = (len(critical_using_defaults) / total_critical) * 100
            else:
                default_percent = 0
                
            # Skip this disease if too many critical fields are missing or using defaults
            # Threshold: if more than 50% of critical fields are missing or using defaults
            if len(critical_missing) > 0 or default_percent > 50:
                reason = "missing critical fields" if critical_missing else "using too many default values"
                skipped_diseases.append(f"{disease_display_map[disease_type]} ({reason})")
                continue
            
            # Try to make prediction with available data
            try:
                result, error = predictor.predict(disease_type, validated_data)
                
                if error:
                    skipped_diseases.append(disease_display_map[disease_type])
                    continue
                
                # Add to successful predictions
                if result and isinstance(result, dict):
                    predictions[disease_type] = {
                        'result': result,
                        'validated_data': validated_data,
                        'missing_fields': missing_fields,
                        'using_defaults': critical_using_defaults
                    }
                    
                    # Save prediction to database if user is logged in
                    if 'username' in st.session_state:
                        save_prediction(
                            st.session_state['username'],
                            disease_type,
                            validated_data,
                            result['prediction'],
                            result.get('prediction_probability', 0.0)
                        )
            except Exception as e:
                skipped_diseases.append(disease_display_map[disease_type])
                logging.error(f"Error predicting {disease_type}: {str(e)}")
                continue
        
        # Display prediction results
        if predictions:
            st.success(f"Successfully made predictions for {len(predictions)} disease types!")
            
            # Create tabs for each successful prediction
            pred_tabs = st.tabs([disease_display_map[disease] for disease in predictions.keys()])
            
            for i, (disease_type, prediction_data) in enumerate(predictions.items()):
                with pred_tabs[i]:
                    # Display prediction result
                    result = prediction_data['result']
                    show_prediction_result(result, disease_type)
                    
                    # Display extracted data used for prediction
                    st.subheader("Parameters Used")
                    
                    # Get feature info for better display
                    feature_info = predictor.get_feature_info(disease_type)
                    validated_data = prediction_data['validated_data']
                    missing_fields = prediction_data['missing_fields']
                    using_defaults = prediction_data.get('using_defaults', [])
                    
                    # Create two columns for better layout
                    col1, col2 = st.columns(2)
                    use_col1 = True
                    
                    # Show critical fields first, then others
                    critical_fields = critical_fields_by_disease.get(disease_type, [])
                    sorted_features = sorted(feature_info['features'], key=lambda x: (x not in critical_fields, x))
                    
                    for feature in sorted_features:
                        feature_in_data = False
                        feature_key = None
                        
                        # Find feature in validated_data (case-insensitive)
                        for key in validated_data:
                            if key.lower() == feature.lower():
                                feature_in_data = True
                                feature_key = key
                                break
                                
                        if feature_in_data:
                            current_col = col1 if use_col1 else col2
                            use_col1 = not use_col1
                            
                            with current_col:
                                # Get feature type and label
                                feature_type = feature_info['types'].get(feature, 'numeric')
                                feature_label = feature_info.get('labels', {}).get(feature, feature)
                                
                                value = validated_data[feature_key]
                                
                                # Check if this is a default value
                                is_default = any(default.lower() == feature.lower() for default in using_defaults)
                                prefix = "🔄 " if is_default else ""  # Add icon for default values
                                
                                # Convert numeric categorical values to human-readable text
                                if feature_type == 'categorical' and disease_type in predictor.reverse_mappings:
                                    if feature in predictor.reverse_mappings[disease_type]:
                                        try:
                                            numeric_value = int(float(value))
                                            if numeric_value in predictor.reverse_mappings[disease_type][feature]:
                                                display_value = predictor.reverse_mappings[disease_type][feature][numeric_value]
                                                if is_default:
                                                    st.warning(f"{prefix}**{feature_label}**: {display_value} ({numeric_value}) - Default value")
                                                else:
                                                    st.info(f"{prefix}**{feature_label}**: {display_value} ({numeric_value})")
                                                continue
                                        except (ValueError, TypeError):
                                            pass
                                
                                # For numeric values or if conversion failed
                                if isinstance(value, (int, float)):
                                    if is_default:
                                        st.warning(f"{prefix}**{feature_label}**: {value:.2f} - Default value")
                                    else:
                                        st.info(f"{prefix}**{feature_label}**: {value:.2f}")
                                else:
                                    if is_default:
                                        st.warning(f"{prefix}**{feature_label}**: {value} - Default value")
                                    else:
                                        st.info(f"{prefix}**{feature_label}**: {value}")
                                        
                    # Add disclaimer if using defaults
                    if using_defaults:
                        st.warning(f"⚠️ This prediction uses {len(using_defaults)} default values which may affect accuracy.")
        else:
            st.warning("Could not make any disease predictions with the data extracted from this PDF.")
        
        # Display skipped diseases
        if skipped_diseases:
            with st.expander("Diseases that couldn't be predicted due to missing data"):
                for disease in skipped_diseases:
                    st.write(f"• {disease}")
        
        # Clean up temporary file
        try:
            os.unlink(tmp_file_path)
        except Exception as e:
            print(f"Error removing temporary file: {str(e)}")
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        # Print detailed error for debugging
        print(f"Detailed error: {str(e)}")
        import traceback
        traceback.print_exc()

def show_prediction_page():
    """Show the prediction page with multiple disease predictions."""
    st.header("Medical Prediction")
    
    # Initialize the predictor
    from components.prediction import DiseasePredictor
    predictor = DiseasePredictor()
    
    # Create main tabs for prediction methods
    method_tabs = st.tabs(["Upload Medical Report", "SmartMatch Predict", "Manual Input"])
    
    # PDF Upload Tab
    with method_tabs[0]:
        st.subheader("Upload Medical Report")
        st.write("Upload your medical report (PDF) for disease risk assessment.")
        st.write("")
        
        # Disease selection dropdown with updated disease list
        disease_map = {
            "Heart Disease": 'heart',
            "Diabetes": 'diabetes',
            "Stroke": 'stroke',
            "Liver Disease": 'liver',
            "Kidney Disease": 'kidney',
            "Parkinson's Disease": 'parkinsons'
        }
        selected_disease = st.selectbox(
            "Select Disease for Prediction",
            list(disease_map.keys())
        )
        disease_type = disease_map[selected_disease]
        
        st.write("")
        uploaded_file = st.file_uploader("Upload Medical Report (PDF)", type=['pdf'])
        if uploaded_file is not None:
            process_pdf(predictor, disease_type, uploaded_file)
    
    # SmartMatch Tab
    with method_tabs[1]:
        st.subheader("SmartMatch Predict")
        st.write("""Upload a medical report to get predictions for all compatible disease models. 
                  SmartMatch automatically detects what diseases can be predicted based on the available data in your report.""")
        st.write("")
        
        uploaded_file = st.file_uploader("Upload Medical Report (PDF)", type=['pdf'], key="smartmatch_uploader")
        if uploaded_file is not None:
            process_smartmatch(uploaded_file)
    
    # Manual Input Tab
    with method_tabs[2]:
        st.subheader("Manual Input")
        
        # Get disease type and features
        selected_disease = st.selectbox(
            "Select Disease",
            list(disease_map.keys()),
            key="manual_disease"
        )
        disease_type = disease_map[selected_disease]
        
        # Get feature information
        feature_info = predictor.get_feature_info(disease_type)
        
        if feature_info:
            show_manual_input_form(disease_type)

def show_prediction_result(result: dict, disease_name: str):
    """Helper function to display prediction results consistently"""
    # Get probability value, checking both possible keys
    prob = result.get('probability', result.get('prediction_probability', 0))
    prob_formatted = f"{prob*100:.1f}%" if prob else "N/A"
    
    if result['prediction']:
        st.error(f"High Risk of {disease_name}")
    else:
        st.success(f"Low Risk of {disease_name}")
    
    with st.expander("View Detailed Results"):
        st.write("Result:", "High Risk" if result['prediction'] else "Low Risk")
        st.write(f"Confidence: {prob_formatted}")
        details = {
            'Prediction': 'High Risk' if result['prediction'] else 'Low Risk',
            'Confidence': prob_formatted,
            'Time': datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        }
        
        if 'input_data' in result:
            details['Input Data'] = result['input_data']
            
        st.json(details)

def show_visualization_dashboard():
    """Show visualization dashboard with multiple visualization options"""
    from .visualization import Visualizer
    
    # Initialize visualizer
    visualizer = Visualizer()
    
    # Use the new tabbed interface from the visualizer
    visualizer.show_visualization_interface()

def show_dashboard():
    """Main dashboard function integrating all components"""
    # Define disease map at the function level to be accessible in all sections
    disease_map = {
        "Diabetes": "diabetes",
        "Heart Disease": "heart",
        "Stroke": "stroke",
        "Kidney Disease": "kidney",
        "Liver Disease": "liver",
        "Parkinson's Disease": "parkinsons",
        "Brain Tumor": "brain_tumor"
    }
    
    # Optional components - will be imported only if available
    try:
        from components.chatbot import show_chatbot
        CHATBOT_AVAILABLE = True
    except ImportError:
        CHATBOT_AVAILABLE = False

    try:
        from components.resource import show_resource
        RESOURCE_AVAILABLE = True
    except ImportError:
        RESOURCE_AVAILABLE = False

    # Sidebar configuration
    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",
            options=["Predict", "Visualize", "Chat Bot" if CHATBOT_AVAILABLE else None, "Resources" if RESOURCE_AVAILABLE else None, "Settings", "Statistics", "Logout"],
            icons=["heart-pulse", "bar-chart", "robot", "book", "gear", "graph-up", "box-arrow-right"],
            menu_icon="list",
            styles={
                "container": {"padding": "5px", "background-color": "#0e1117"},
                "icon": {"color": "", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left"},
                "nav-link-selected": {"background-color": "#262730", "color": "white"},
            }
        )

    # Main content area - Header is consistent across all pages
    col1, col2,col3 = st.columns([1, 8,3])
    with col1:
        st_lottie(animation, height=100, width=50, speed=10)
    with col2:
        st.title("HealthSphere")
        username = st.session_state.get("username", "Guest")
        
    with col3:
        st.write(f"Welcome, **{username}**!")
        
    # Handle navigation options
    if selected == "Predict":
        show_prediction_page()

    elif selected == "Visualize":
        show_visualization_dashboard()

    elif selected == "Chat Bot":
        show_chatbot()

    elif selected == "Resources":
        show_resource()

    elif selected == "Settings":
        show_settings()

    elif selected == "Statistics":
        show_stats()
        
    elif selected == "Logout":
        # Clear session state and redirect to login page
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.logged_in = False
        st.rerun()

if __name__ == "__main__":
    show_dashboard()
