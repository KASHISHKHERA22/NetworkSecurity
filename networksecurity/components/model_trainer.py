import sys
import os 
import pandas as pd
import numpy as np 
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging  
from networksecurity.entity.config_entity import DataTransformationConfig, ModelTrainerConfig
from networksecurity.utils.main_utils.utils import save_object, load_numpy_array,load_object,save_numpy_array_data
from networksecurity.entity.artifacts_entity import DataTransformationArtifacts, ModelTrainerArtifacts, ClassificationMetricArtifact
from networksecurity.utils.ml_utils.metrics.model_classification import get_classification_score
from networksecurity.utils.ml_utils.models.estimator import NetworkModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)


class  ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifcats:DataTransformationArtifacts):
            try:
                 
                self.model_trainer_config = model_trainer_config
                self.data_transformation_artifacts = data_transformation_artifcats
            except Exception as e:
                raise NetworkSecurityException(e, sys)
    
    def train_model(self,X_train,y_train,X_test,y_test):
        try:
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
            params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
            
        }
            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def initailte_model_trainer(self)->ModelTrainerArtifacts:
        try:

            train_file_path = self.data_transformation_artifacts.transformed_train_file_path
            test_file_path = self.data_transformation_artifacts.transformed_test_file_path

            train_arr = load_numpy_array(file_path=train_file_path)
            test_arr = load_numpy_array(file_path=test_file_path) 

            X_train,y_train,X_test,y_test = (
                     train_arr[:,:-1],
                     train_arr[:,-1],
                     test_arr[:,:-1],
                     test_arr[:,-1]
              )
            model_trainer_artifact = self.train_model(X_train,y_train,X_test,y_test)

            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)


