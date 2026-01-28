# HealthSphere

HealthSphere is a comprehensive health prediction platform that integrates multiple disease prediction models with data visualization capabilities and a medical chatbot.

## Features

- **Predictive Models**: Predict risk for multiple diseases including:
  - Heart Disease
  - Diabetes
  - Liver Disease
  - Kidney Disease
  - Stroke
  - Parkinson's Disease

- **Data Input Methods**:
  - Manual data entry with guided input forms
  - PDF medical report extraction with smart feature recognition
  
- **User Dashboard**:
  - Personalized user accounts
  - Prediction history tracking
  - Advanced data visualization

- **Medical Chatbot**:
  - Health information and guidance
  - Answers to common health questions

## PDF Extraction Capabilities

The PDF extraction system has been enhanced to handle a wide variety of medical report formats:

- **Multiple Extraction Methods**:
  - Text-based extraction using PyPDF2
  - Advanced extraction with pdfplumber (handles tables)
  - OCR for image-based PDFs using Tesseract

- **Smart Field Recognition**:
  - Comprehensive field mappings for all disease types
  - Fuzzy matching for field names with variations
  - Context-aware value extraction with units recognition

- **Robust Processing**:
  - Handles missing or partial data
  - Works with diverse PDF layouts and formats
  - Extracts tabular data and structured reports

## Utilities for Testing and Development

- **Test PDF Extraction**: Use `test_pdf_extraction.py` to evaluate extraction quality
  ```
  python test_pdf_extraction.py path/to/pdf_file.pdf --disease heart --verbose
  ```

- **Generate Test PDFs**: Create sample medical reports for testing
  ```
  python generate_test_pdfs.py --disease diabetes --count 5
  ```

## Visualization Module

The enhanced visualization module now offers a tabbed interface with two main sections:

### Info Tab

Provides informational visualizations about your health data:

- **Prediction History**: View history of predictions for specific diseases
- **Parameter Trends**: Track how specific health parameters change over time
- **Parameter Correlation**: Analyze relationships between different health metrics
- **Overall Health Analysis**: Get a comprehensive view of your health status
- **Trend Analysis**: See how disease risks change over time
- **Feature Comparison**: Compare multiple health parameters simultaneously

### Visualize Tab

Create advanced, customizable visualizations for deeper analysis:

- **2D Line Plot**: Track parameter changes with line plots
- **2D Scatter Plot**: Analyze relationships between parameters
- **3D Parameter Analysis**: Visualize relationships between three parameters
- **3D Scatter Plot**: Create interactive 3D scatter plots
- **3D Surface Plot**: Generate surface plots for complex data
- **3D Line Plot**: Track multiple parameters in 3D space
- **3D Mesh Plot**: Create mesh visualizations for complex relationships

### Usage

1. Select the appropriate tab (Info or Visualize)
2. Choose the disease you want to analyze
3. Select a visualization type
4. Customize parameters as needed (especially in the Visualize tab)
5. Interact with the generated plot to explore your data

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python app.py
   ```

3. Access the web interface at `http://localhost:8501`

4. Log in to your account
5. Navigate to the dashboard
6. Use the navigation menu to access the visualization component
7. Start exploring your health data through interactive visualizations

## Technology Stack

- **Backend**: Python, Streamlit
- **Database**: MongoDB
- **ML Libraries**: Scikit-learn, TensorFlow
- **PDF Processing**: PyPDF2, pdfplumber, pytesseract, pdf2image
- **Visualization**: Plotly, Streamlit components 

## Technical Requirements

- Python 3.7+
- Streamlit
- Plotly
- MongoDB
- PyMongo 