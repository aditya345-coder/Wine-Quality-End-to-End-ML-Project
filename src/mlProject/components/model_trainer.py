from lightgbm import LGBMClassifier
import pandas as pd
import os

# from sklearn.base import accuracy_score
from mlProject import logger
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Load train and test data
        logger.info("Loading training and test data")
        X_train = pd.read_csv(self.config.X_train_data_path)
        X_test = pd.read_csv(self.config.X_test_data_path)
        y_train = pd.read_csv(self.config.y_train_data_path)
        y_test = pd.read_csv(self.config.y_test_data_path)
        
        # Extract the target variable from the training data
        all_features = X_train.columns.tolist()
        
        # Wrap ColumnTransformer in a pipeline with MinMaxScaler
        preprocessing_pipeline = Pipeline([
            ('minmax_scaler', MinMaxScaler())
        ])

        # Apply pipeline to training and test sets
        X_train_processed = preprocessing_pipeline.fit_transform(X_train)
        X_test_processed = pd.DataFrame(preprocessing_pipeline.transform(X_test), columns=all_features)

        X_test_processed.to_csv(os.path.join(self.config.root_dir, "X_test_processed.csv"), index=False)
        y_test.to_csv(os.path.join(self.config.root_dir, "y_test.csv"), index=False)
        # Save the preprocessing pipeline
        joblib.dump(preprocessing_pipeline, os.path.join(self.config.root_dir, self.config.scaler_name))

        # Apply SMOTE to balance the training data
        smote = SMOTE(random_state=42, k_neighbors=2)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
        logger.info(f"Resampled training data shape: {X_train_resampled.shape}")
        
        # Train the RandomForestClassifier
        model = RandomForestClassifier(
            max_depth=self.config.max_depth,
            class_weight=self.config.class_weight,
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
        )
        
        model.fit(X_train_resampled, y_train_resampled)
        logger.info("Model training completed")
        logger.info(f"Model parameters: {model.get_params()}")
        
        # Save the trained model
        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))