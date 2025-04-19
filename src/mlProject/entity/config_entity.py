from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path



@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict



@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path



@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    X_train_data_path: Path
    y_train_data_path: Path
    X_test_data_path: Path
    y_test_data_path: Path
    model_name: str
    scaler_name: str
    max_depth: int
    class_weight: str
    n_estimators: int
    random_state: int
    target_column: str



@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    X_test_data_path: Path
    y_test_data_path: Path
    model_path: Path
    scaler_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str