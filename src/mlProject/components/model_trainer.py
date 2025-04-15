import pandas as pd
import os
from mlProject import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
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
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Split features and target
        X_train = train_data.drop([self.config.target_column], axis=1)
        X_test = test_data.drop([self.config.target_column], axis=1)
        y_train = train_data[self.config.target_column]
        y_test = test_data[self.config.target_column]

        # Define skewed and non-skewed features
        skewed_features = ['volatile acidity', 'citric acid', 'residual sugar', 'chlorides',  
                           'free sulfur dioxide', 'total sulfur dioxide', 'sulphates', 'alcohol']
        all_features = X_train.columns.tolist()
        other_features = [col for col in all_features if col not in skewed_features]

        # ColumnTransformer: Apply PowerTransformer on skewed features and passthrough others
        column_transformer = ColumnTransformer(transformers=[
            ('power', PowerTransformer(method='yeo-johnson'), skewed_features),
            ('passthrough', 'passthrough', other_features)
        ])

        # Wrap ColumnTransformer in a pipeline with MinMaxScaler
        preprocessing_pipeline = Pipeline([
            ('transformer', column_transformer),
            ('minmax_scaler', MinMaxScaler())
        ])

        # Apply pipeline to training and test sets
        X_train_processed = preprocessing_pipeline.fit_transform(X_train)
        X_test_processed = preprocessing_pipeline.transform(X_test)

        # Save the preprocessing pipeline
        joblib.dump(preprocessing_pipeline, os.path.join(self.config.root_dir, self.config.scaler_name))

        # Apply SMOTE to balance the training data
        smote = SMOTE(random_state=42, k_neighbors=2)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

        # Train the RandomForestClassifier
        model = RandomForestClassifier(
            max_depth=self.config.max_depth,
            class_weight=self.config.class_weight,
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state
        )
        model.fit(X_train_resampled, y_train_resampled)

        # Save the trained model
        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))