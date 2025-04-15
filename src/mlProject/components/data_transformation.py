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
        if score <= 4:
            return 'Low'
        elif score <= 6:
            return 'Medium'
        else:
            return 'High'

    def train_test_spliting(self):
        """Method to load data and split into train/test sets."""
        logger.info("Loading data")
        # Load data
        data = pd.read_csv(self.config.data_path)

        # Map the quality score to categorical labels
        data['quality_label'] = data['quality'].apply(self._map_quality)

        # Drop the original quality column
        data.drop(columns='quality', inplace=True)

        # Split the data into training and test sets
        train, test = train_test_split(data, test_size=0.2, random_state=42)

        # Save the transformed datasets
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Mapped wine quality and split data into training and test sets")
        logger.info(f"Train shape: {train.shape}")
        logger.info(f"Test shape: {test.shape}")

        print(train.shape)
        print(test.shape)

        