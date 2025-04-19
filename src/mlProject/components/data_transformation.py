import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def _map_quality(self, score):
        """Private method to map wine quality to Low/Medium/High."""
        if score <= 5:
            return 'Low'
        else:
            return 'High'
        
    def clip_outliers_iqr(self,data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return data[column].clip(lower=lower, upper=upper)

    def train_test_spliting(self):
        """Method to load data and split into train/test sets."""
        logger.info("Loading data")
        # Load data
        data = pd.read_csv(self.config.data_path)

        columns = list(data.columns)
        
        for col in columns:
            data[col] = self.clip_outliers_iqr(data, col)
        # Map the quality score to categorical labels
        data['quality_label'] = data['quality'].apply(self._map_quality)

        # Drop the original quality column
        data.drop(columns=['quality', 'Id'], inplace=True)

        # Split the data into training and test sets
        X = data.drop(columns=['quality_label'])
        y = data['quality_label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
         
        # Save the transformed datasets
        X_train.to_csv(os.path.join(self.config.root_dir, "X_train.csv"), index=False)
        y_train.to_csv(os.path.join(self.config.root_dir, "y_train.csv"), index=False)
        X_test.to_csv(os.path.join(self.config.root_dir, "X_test.csv"), index=False)
        y_test.to_csv(os.path.join(self.config.root_dir, "y_test.csv"), index=False)
        
        logger.info("Mapped wine quality and split data into training and test sets")
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"y_train shape: {y_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}")
        logger.info(f"y_test shape: {y_test.shape}")

        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)

        