# Wine-Quality-End-to-End-ML-Project


# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/aditya345-coder/Wine-Quality-End-to-End-ML-Project
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mlproj python=3.11 -y
```

```bash
conda activate mlproj
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

```bash
python main.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up your local host and port
```



## [MLflow](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

### [DagsHub](https://dagshub.com/)

import dagshub
dagshub.init(repo_owner='neuralninja01', repo_name='Wine-Quality-End-to-End-ML-Project', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/neuralninja01/Wine-Quality-End-to-End-ML-Project.mlflow

export MLFLOW_TRACKING_USERNAME=neuralninja01
```

## About MLflow 
MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & tagging your model

## Deployment

You can check the deployed version [here](https://wine-quality-end-to-end-ml-project.onrender.com/).


