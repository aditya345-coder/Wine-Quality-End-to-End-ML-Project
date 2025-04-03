# 🍷 Wine Quality Prediction – End-to-End ML Project  

## 📌 Overview  
This project is an end-to-end machine learning system that predicts wine quality based on its characteristics. It includes data preprocessing, model training, experiment tracking with MLflow, and deployment via Flask.  

## 🚀 How to Run  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/aditya345-coder/Wine-Quality-End-to-End-ML-Project.git
cd Wine-Quality-End-to-End-ML-Project
```

### 2️⃣ Create and Activate Virtual Environment  
```bash
python -m venv venv
```
#### Windows  
```bash
venv\Scripts\activate
```
#### macOS/Linux  
```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application  
```bash
python main.py   # Model training
python app.py    # Run Flask web app
```

### 5️⃣ Access Locally  
After running `app.py`, open your browser and visit:  
```
http://127.0.0.1:<PORT>
```
(Replace `<PORT>` with the actual port displayed in the terminal.)  

## 📊 Experiment Tracking with MLflow  
### Start MLflow UI  
```bash
mlflow ui
```
Access MLflow dashboard at:  
```
http://127.0.0.1:5000
```

### Track Experiments on DagsHub  
Initialize tracking in Python:  
```python
import dagshub
dagshub.init(repo_owner='neuralninja01', repo_name='Wine-Quality-End-to-End-ML-Project', mlflow=True)

import mlflow
with mlflow.start_run():
    mlflow.log_param('parameter_name', 'value')
    mlflow.log_metric('metric_name', 1)
```
Set environment variables for tracking:  
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/neuralninja01/Wine-Quality-End-to-End-ML-Project.mlflow
export MLFLOW_TRACKING_USERNAME=neuralninja01
```

## 📦 Deployment  
The project is deployed and accessible here:  
🔗 **[Live Demo](https://wine-quality-end-to-end-ml-project.onrender.com/)**  

---

This version improves clarity, formatting, and readability while making it more structured and engaging. Let me know if you need any modifications! 🚀
