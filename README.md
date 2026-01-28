# HealthSphere
HealthSphere is an intelligent healthcare web application built with Python and Streamlit that leverages machine learning to predict the risk of multiple diseases, analyze medical reports, visualize health insights, and assist users through an interactive chatbot.

This project demonstrates real-world skills in ML, data processing, UI development, and system design.

✨ Key Features
🧠 Disease Prediction (Machine Learning)

Predicts the likelihood of:

Heart Disease

Diabetes

Liver Disease

Kidney Disease

Stroke

Parkinson’s Disease

Models are trained using scikit-learn and saved with joblib for efficient inference.

📊 Interactive Health Dashboard

Dynamic charts using Plotly

User-friendly UI with Streamlit

Visual insights for better understanding of health metrics

📄 Medical Report Analysis

Upload and extract text from medical PDFs

Supports scanned reports via OCR (pytesseract)

Uses:

PyPDF2

pdfplumber

pdf2image

pytesseract

🤖 Healthcare Chatbot

An integrated chatbot to assist users with basic health-related queries and guidance.

🗄️ Database Integration

MongoDB support using PyMongo

Can store user inputs, predictions, and history (optional feature)

🛠️ Tech Stack

Language: Python

Web Framework: Streamlit

Machine Learning: Scikit-learn

Visualization: Plotly

Database: MongoDB

PDF & OCR: PyPDF2, pdfplumber, pytesseract

Model Handling: joblib

📂 Project Structure
Healthsphere/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
│
├── components/             # Core functionality modules
│   ├── chatbot.py
│   ├── dashboard.py
│   ├── database.py
│   ├── pdf.py
│   ├── statistics.py
│   ├── train_models.py
│   ├── utils.py
│   └── visualization.py
│
├── dataset/                # CSV datasets used for training
├── models/                 # Trained ML models, scalers, features
├── reports/                # Report generation files
├── Animation/              # UI assets (GIFs, Lottie files, etc.)

⚙️ Installation & Running Locally
1. Clone the repository
git clone https://github.com/your-username/HealthSphere.git
cd HealthSphere

2. (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

3. Install dependencies
pip install -r requirements.txt

4. Run the application
streamlit run app.py


The app will open in your browser at:

http://localhost:8501

🎯 Project Objectives

Apply machine learning to real-world healthcare problems

Build an interactive web application using Streamlit

Practice modular project structure and clean code

Combine ML, UI, data visualization, and document processing in one system

📌 Future Enhancements

User authentication (login/signup)

Deployment to Streamlit Cloud / AWS

Improved chatbot with LLM integration

Mobile-responsive UI

Doctor/Admin dashboards

More disease prediction models
