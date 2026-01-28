import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def create_model_directory():
    """Create models directory if it doesn't exist"""
    # Create models directory at the right location
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    return models_dir

def get_dataset_path(filename):
    """Return the path to a dataset file"""
    # Get path to dataset folder relative to this script
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'dataset', filename)
    return dataset_path

def load_and_preprocess_diabetes():
    """Load and preprocess updated diabetes dataset"""
    df = pd.read_csv(get_dataset_path('diabetes_updated.csv'))
    X = df.drop('outcome', axis=1)
    y = df['outcome']
    return X, y

def load_and_preprocess_heart():
    """Load and preprocess updated heart disease dataset"""
    df = pd.read_csv(get_dataset_path('heart_updated.csv'))
    
    # Define the features and target
    X = df.drop('Target', axis=1)
    y = df['Target']
    return X, y

def load_and_preprocess_stroke():
    """Load and preprocess updated stroke dataset"""
    df = pd.read_csv(get_dataset_path('stroke_updated.csv'))
    
    # Drop id column if present
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    # Process categorical columns if needed
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    # Handle any missing values
    df = df.fillna(df.mean())
    
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    return X, y

def load_and_preprocess_kidney():
    """Load and preprocess updated kidney disease dataset"""
    df = pd.read_csv(get_dataset_path('kidney_updated.csv'))
    
    # Print columns to check what's available
    print("\nKidney Disease Dataset Columns:", df.columns.tolist())
    
    # Process categorical columns
    categorical_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                categorical_columns.append(col)
            except:
                print(f"Error encoding column: {col}")
    
    print("\nProcessed categorical columns:", categorical_columns)
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Updated target column name based on the actual CSV headers
    target_col = 'Chronic Kidney Disease'
    
    X = df.drop(['Patient ID', target_col], axis=1)
    y = df[target_col]
    return X, y

def load_and_preprocess_liver():
    """Load and preprocess updated liver disease dataset"""
    df = pd.read_csv(get_dataset_path('liver_updated.csv'))
    
    # Process Gender if it's categorical
    if 'Gender' in df.columns and df['Gender'].dtype == 'object':
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Updated target column name based on the actual CSV headers
    target_col = 'Liver Disease'
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y

def load_and_preprocess_parkinsons():
    """Load and preprocess updated Parkinson's dataset"""
    df = pd.read_csv(get_dataset_path('parkinsons_updated.csv'))
    
    # Identify and remove non-feature columns
    non_feature_cols = ['Patient Name']
    
    # Updated target column name based on the actual CSV headers
    target_col = "Parkinson's Disease"
    
    # Drop non-feature columns
    drop_cols = non_feature_cols + [target_col]
    X = df.drop(drop_cols, axis=1)
    y = df[target_col]
    return X, y

def train_and_save_model(X, y, model_name, models_dir):
    """Train model and save it"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Print feature names for reference
    print(f"\nFeatures for {model_name}:", X.columns.tolist())
    
    # Train model with better parameters
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    
    # Train final model
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel: {model_name}")
    print(f"Test Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and scaler
    joblib.dump(model, os.path.join(models_dir, f'{model_name}_model.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, f'{model_name}_scaler.joblib'))
    
    # Save feature names for reference
    with open(os.path.join(models_dir, f'{model_name}_features.txt'), 'w') as f:
        f.write(','.join(X.columns.tolist()))
    
    return X.columns.tolist()

def main():
    print("Starting model training with updated datasets...")
    models_dir = create_model_directory()
    print(f"Models will be saved to: {models_dir}")
    
    # Dictionary to store feature lists for each model
    feature_lists = {}
    
    try:
        # Train diabetes model
        print("\nTraining Diabetes Model...")
        X, y = load_and_preprocess_diabetes()
        feature_lists['diabetes'] = train_and_save_model(X, y, 'diabetes', models_dir)
    except Exception as e:
        print(f"Error training diabetes model: {str(e)}")
    
    try:
        # Train heart disease model
        print("\nTraining Heart Disease Model...")
        X, y = load_and_preprocess_heart()
        feature_lists['heart'] = train_and_save_model(X, y, 'heart', models_dir)
    except Exception as e:
        print(f"Error training heart model: {str(e)}")
    
    try:
        # Train stroke model
        print("\nTraining Stroke Model...")
        X, y = load_and_preprocess_stroke()
        feature_lists['stroke'] = train_and_save_model(X, y, 'stroke', models_dir)
    except Exception as e:
        print(f"Error training stroke model: {str(e)}")
    
    try:
        # Train kidney disease model
        print("\nTraining Kidney Disease Model...")
        X, y = load_and_preprocess_kidney()
        feature_lists['kidney'] = train_and_save_model(X, y, 'kidney', models_dir)
    except Exception as e:
        print(f"Error training kidney model: {str(e)}")
    
    try:
        # Train liver disease model
        print("\nTraining Liver Disease Model...")
        X, y = load_and_preprocess_liver()
        feature_lists['liver'] = train_and_save_model(X, y, 'liver', models_dir)
    except Exception as e:
        print(f"Error training liver model: {str(e)}")
    
    try:
        # Train Parkinson's model
        print("\nTraining Parkinson's Disease Model...")
        X, y = load_and_preprocess_parkinsons()
        feature_lists['parkinsons'] = train_and_save_model(X, y, 'parkinsons', models_dir)
    except Exception as e:
        print(f"Error training Parkinson's model: {str(e)}")
    
    print("\nAll models have been trained and saved!")
    
    # Print summary of features for all models
    print("\nFeature Summary for All Models:")
    for disease, features in feature_lists.items():
        print(f"\n{disease.upper()} - {len(features)} features:")
        print(", ".join(features))

if __name__ == "__main__":
    main() 