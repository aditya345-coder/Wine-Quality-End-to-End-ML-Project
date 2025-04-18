# 🍷 Wine Quality Prediction – End-to-End ML Project  

## 📌 Overview  
This project aims to predict the quality of red wine based on various physicochemical attributes using machine learning. The project follows a complete end-to-end ML pipeline, from data preprocessing and modeling to deployment using a Flask web application.

## Project Structure

```
|   
+---config
|       config.yaml                # Configuration settings for the project
|       
+---notebook
|       01_data_ingestion.ipynb     # Notebook for data collection and loading
|       02_data_validation.ipynb    # Notebook for data validation and preprocessing checks
|       03_data_transformation.ipynb # Notebook for feature engineering and transformation
|       04_model_trainer.ipynb      # Notebook for training machine learning models
|       05_model_evaluation.ipynb   # Notebook for evaluating model performance
|       trials.ipynb                 # Experimental notebooks for testing various approaches
|       
+---src
|   \---mlProject
|       |   __init__.py              # Initialization file for the package
|       |   
|       +---components
|       |       data_ingestion.py     # Handles data collection and preprocessing
|       |       data_transformation.py # Transforms raw data into a suitable format
|       |       data_validation.py    # Ensures data integrity and validity
|       |       model_evaluation.py   # Evaluates the performance of trained models
|       |       model_trainer.py      # Trains machine learning models
|       |       __init__.py           
|       |       
|       +---config
|       |       configuration.py      # Configuration settings management
|       |       __init__.py
|       |       
|       +---constants
|       |       __init__.py           # Stores project-wide constant values
|       |       
|       +---entity
|       |       config_entity.py      # Defines configuration data structures
|       |       __init__.py
|       |       
|       +---pipeline
|       |       prediction.py         # Prediction pipeline
|       |       stage_01_data_ingestion.py # Data ingestion pipeline stage
|       |       stage_02_data_validation.py # Data validation pipeline stage
|       |       stage_03_data_transformation.py # Data transformation pipeline stage
|       |       stage_04_model_trainer.py # Model training pipeline stage
|       |       stage_05_model_evaluation.py # Model evaluation pipeline stage
|       |       __init__.py           
|       |       
|       \---utils
|               common.py              # Utility functions for various tasks
|               __init__.py
|               
+---templates
|       index.html                    # HTML template for web UI
|
|   .gitignore                         # Specifies files to be ignored by Git
|   app.py                             # Streamlit app to interact with the recommendation system
|   Dockerfile                         # Docker setup file
|   LICENSE                            # License details
|   main.py                            # Entry point for training the models
|   params.yaml                        # Hyperparameters configuration file
|   README.md                          # Project documentation
|   requirements.txt                    # Dependencies required for the project
|   schema.yaml                        # Data schema definition
|   setup.py                           # Project setup script
|   template.py                        # Template management file
|   test.py                            # Script for testing components
```

---
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

---

## Demo

---

## 📦 Deployment  
The project is deployed and accessible here:  
🔗 **[Live Demo](https://wine-quality-end-to-end-ml-project.onrender.com/)**  

---
