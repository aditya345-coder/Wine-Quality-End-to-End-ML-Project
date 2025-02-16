import pandas as pd
import os
from mlProject import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        X_train = train_data.drop([self.config.target_column], axis=1)
        X_test = test_data.drop([self.config.target_column], axis=1)
        y_train = train_data[[self.config.target_column]].squeeze()
        y_test = test_data[[self.config.target_column]].squeeze()

        scaler=StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        joblib.dump(scaler, os.path.join(self.config.root_dir, self.config.scaler_name))

        rfc=RandomForestClassifier(max_depth=self.config.max_depth, n_estimators=self.config.n_estimators, random_state=self.config.random_state)
        rfc.fit(X_train, y_train)

        joblib.dump(rfc, os.path.join(self.config.root_dir, self.config.model_name))

