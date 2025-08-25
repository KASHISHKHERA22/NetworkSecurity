from dataclasses import dataclass
from typing import Optional
from typing import Union
import numpy as np  
@dataclass
class dataIngestionArtifact:
    trained_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifacts:
    validation_status:bool
    valid_train_file_path:str
    valid_test_file_path:str
    invalid_train_file_path: Optional[str]
    invalid_test_file_path:Optional[str]
    drift_report_file_path:str

@dataclass
class DataTransformationArtifacts:
    transformed_object_file_path:str
    transformed_train_file_path:str
    transformed_test_file_path:str

@dataclass
class ClassificationMetricArtifact:
    f1_score: Union[float, np.floating]
    precision_score: Union[float, np.floating]
    recall_score: Union[float, np.floating]
    accuracy_score: Union[float, np.floating]
@dataclass
class ModelTrainerArtifacts:
    trained_model_file_path:str
    train_metric_artifact:ClassificationMetricArtifact
    test_metric_artifact:ClassificationMetricArtifact
    