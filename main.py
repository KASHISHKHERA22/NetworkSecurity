from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecurityException
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
    except Exception as e:
        logging.info("An error occurred in the main block.")
        raise NetworkSecurityException(e, sys)