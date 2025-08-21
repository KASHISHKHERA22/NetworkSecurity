from networksecurity.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig
from networksecurity.entity.artifacts_entity import dataIngestionArtifact, DataValidationArtifacts
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import read_yaml_file,write_yaml_file
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split    
from scipy.stats import ks_2samp
from typing import Any
from scipy import stats
import yaml 

class dataValidation:
    def __init__(self,data_ingestion_artifacts:dataIngestionArtifact, data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifacts = data_ingestion_artifacts
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    @staticmethod   
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def validate_number_of_columns(self,dataFrame:pd.DataFrame) -> bool:
        try:
            no_of_columns = len(self._schema_config)
            if len(dataFrame.columns)==no_of_columns:
                return True
            return False
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)  
    
    def detect_dataset_drfit(self,base_df,current_df,threshold=0.05) -> bool:
        try:
            status =True
            report = {}

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist: Any = ks_2samp(d1,d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({column: {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found
                }})
                
            drift_report_file_path = self.data_validation_config.drift_report_dir_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report)
            return status

        except Exception as e:
            raise NetworkSecurityException(e, sys)
    def initiate_data_validation(self)-> DataValidationArtifacts:
        try:
            train_file_path = self.data_ingestion_artifacts.trained_file_path
            test_file_path = self.data_ingestion_artifacts.test_file_path

            train_dataframe = pd.read_csv(train_file_path)
            test_dataframe = pd.read_csv(test_file_path)
            logging.info("Reading train and test data completed")

            status = self.validate_number_of_columns(train_dataframe)
            if not status:
                errorMsg = f"Train dataframe doesn't contain all columns as per schema"
            status = self.validate_number_of_columns(test_dataframe)
            if not status:
                errorMsg = f"Test dataframe doesn't contain all columns as per schema"  

            status = self.detect_dataset_drfit(base_df=train_dataframe,current_df=test_dataframe)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            data_validation_artifacts = DataValidationArtifacts(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_dir_path
            )

            return data_validation_artifacts

        except Exception as e:
            raise NetworkSecurityException(e, sys)