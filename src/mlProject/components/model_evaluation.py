import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, accuracy_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average='weighted', zero_division=1)
        recall = recall_score(actual, pred, average='weighted', zero_division=1)
        f1 = f1_score(actual, pred, average='weighted')
        conf_matrix = confusion_matrix(actual, pred)
        return accuracy, precision, recall, f1, conf_matrix
    


    def log_into_mlflow(self):

        # test_data = pd.read_csv(self.config.test_data_path)
        X_test_processed = pd.read_csv(self.config.X_test_data_path)
        y_test = pd.read_csv(self.config.y_test_data_path)
        model = joblib.load(self.config.model_path)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        with mlflow.start_run():

            predicted_qualities = model.predict(X_test_processed)

            (accuracy, precision, recall, f1, conf_matrix) = self.eval_metrics(y_test, predicted_qualities)
            
            # Saving metrics as local
            scores = {"accuracy":accuracy, "precision":precision, "recall":recall, "f1":f1, "conf_matrix":conf_matrix.tolist()}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("accuracy",accuracy)
            mlflow.log_metric("precision",precision)
            mlflow.log_metric("recall",recall)
            mlflow.log_metric("f1",f1)
            # mlflow.log_metric("conf_matrix",conf_matrix.tolist())
            
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForestClassifierModel")
            else:
                mlflow.sklearn.log_model(model, "model")