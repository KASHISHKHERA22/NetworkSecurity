from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.components.data_validation import DataValidationConfig, dataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.entity.artifacts_entity import DataTransformationArtifacts, DataValidationArtifacts, dataIngestionArtifact
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
import sys  
import os

if __name__ == "__main__":
    try:
        trainpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainpipelineconfig)
        dataingestion = DataIngestion(dataingestionconfig)
        logging.info("Starting data ingestion process.")
        dataingestionartifact = dataingestion.initaite_data_ingestion()
        print(dataingestionartifact)

        data_validation_config = DataValidationConfig(trainpipelineconfig)
        data_validation = dataValidation(dataingestionartifact, data_validation_config)
        data_validation_artifacts=data_validation.initiate_data_validation()    
        logging.info("Starting data transformation process.")

        data_transformation_config = DataTransformationConfig(trainpipelineconfig)
        data_transformation = DataTransformation(data_validation_artifacts,data_transformation_config)
        data_transformation_artifacts = data_transformation.initiate_data_tranformation()
        print(data_transformation_artifacts)
        logging.info("Data transformation is completed.")
    except Exception as e:
        logging.info("An error occurred in the main block.")
        raise NetworkSecurityException(e, sys)