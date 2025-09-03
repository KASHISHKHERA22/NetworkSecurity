import sys
import os
import pandas as pd
import numpy as np
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import dataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.entity.config_entity import DataTransformationConfig, ModelTrainerConfig,DataIngestionConfig,DataValidationConfig,TrainingPipelineConfig
from networksecurity.utils.main_utils.utils import save_object, load_numpy_array,load_object,save_numpy_array_data,evaluate_models
from networksecurity.entity.artifacts_entity import DataTransformationArtifacts, ModelTrainerArtifacts, ClassificationMetricArtifact,DataValidationArtifacts,dataIngestionArtifact
from networksecurity.utils.ml_utils.metrics.model_classification import get_classification_score
from networksecurity.utils.ml_utils.models.estimator import NetworkModel

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self):
        self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
        dataingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
        data_ingestion_artifacts=dataingestion.initaite_data_ingestion()
        return data_ingestion_artifacts
    
    def start_data_validation(self,data_ingestion_artifacts:dataIngestionArtifact):
        data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
        datavalidation = dataValidation(data_ingestion_artifacts=data_ingestion_artifacts,data_validation_config=data_validation_config)
        data_validation_artifacts = datavalidation.initiate_data_validation()
        return data_validation_artifacts
    
    def start_data_transformation(self,data_validation_artifacts:DataValidationArtifacts):
        data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifacts=data_validation_artifacts,data_transformation_config=data_transformation_config)
        data_transformation_artifacts = data_transformation.initiate_data_tranformation()
        return data_transformation_artifacts
    
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifacts)->ModelTrainerArtifacts:
        try:
            self.model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_trainer = ModelTrainer(model_trainer_config= self.model_trainer_config,data_transformation_artifact=data_transformation_artifact)


            model_trainer_artifact = model_trainer.initaite_model_trainer()

            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def run_pipeline(self):
        try:

            data_ingestion_artifacts = self.start_data_ingestion()
            data_validation_artifacts = self.start_data_validation(data_ingestion_artifacts=data_ingestion_artifacts)
            data_transformation_artifacts   = self.start_data_transformation(data_validation_artifacts=data_validation_artifacts)                       
            model_trainer_artifacts = self.start_model_trainer(data_transformation_artifact=data_transformation_artifacts)
            return model_trainer_artifacts
        except Exception as e:      
            raise NetworkSecurityException(e, sys)