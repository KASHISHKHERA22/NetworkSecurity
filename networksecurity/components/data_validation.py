from networksecurity.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig
from networksecurity.entity.artifacts_entity import dataIngestionArtifact, DataValidationArtifacts
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split    
from scipy.stats import ks_2samp
import yaml 

class dataValidation:
    def __init__(self,data_ingestion_artifacts:dataIngestionArtifact, data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifacts = data_ingestion_artifacts
            self.data_validation_config = data_validation_config
            self.schema_file_path = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)