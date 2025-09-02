import sys
import os 
import pandas as pd
import numpy as np 
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging  
from networksecurity.entity.config_entity import DataTransformationConfig, ModelTrainerConfig
from networksecurity.utils.main_utils.utils import save_object, load_numpy_array,load_object,save_numpy_array_data,evaluate_models
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
import mlflow.sklearn
import mlflow
from mlflow.sklearn import log_model
from dagshub.logger import dagshub_logger
from dagshub import logger
from urllib.parse import urlparse
dagshub_logger("KASHISHKHERA22/NetworkSecurity")


os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/KASHISHKHERA22/NetworkSecurity.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="KASHISHKHERA22"
os.environ["MLFLOW_TRACKING_PASSWORD"]="a9576f1b600259e5a912638a90a6050ea6ace6e0"


class  ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifcats:DataTransformationArtifacts):
            try:
                 
                self.model_trainer_config = model_trainer_config
                self.data_transformation_artifacts = data_transformation_artifcats
            except Exception as e:
                raise NetworkSecurityException(e, sys)
    
    def track_mlflow(self,best_model,classificationmetric,model_name:str):
        mlflow.set_tracking_uri("https://dagshub.com/KASHISHKHERA22/NetworkSecurity.mlflow")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            f1_score = classificationmetric.f1_score
            precision_score = classificationmetric.precision_score
            recall_score = classificationmetric.recall_score
            accuracy_score = classificationmetric.accuracy_score

            mlflow.log_metric("F1-Score",f1_score)
            mlflow.log_metric("Precision-Score",precision_score)
            mlflow.log_metric("Recall-Score",recall_score)
            mlflow.log_metric("Accuracy-Score",accuracy_score)
            if tracking_url_type_store != "file":

                 # Register the model
    #             # There are other ways to use the Model Registry, which depends on the use case,
    #             # please refer to the doc for more information:
    #             # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                log_model(best_model, "model", registered_model_name=model_name)
            else:
                log_model(best_model, "model")

    # def track_mlflow(self, best_model, classificationmetric, model_name: str):
        # mlflow.set_tracking_uri("https://dagshub.com/KASHISHKHERA22/NetworkSecurity.mlflow")
        # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # with mlflow.start_run():
        #     f1_score = classificationmetric.f1_score
        #     precision_score = classificationmetric.precision_score
        #     recall_score = classificationmetric.recall_score
        #     accuracy_score = classificationmetric.accuracy_score

        #     mlflow.log_metric("F1-Score", f1_score)
        #     mlflow.log_metric("Precision-Score", precision_score)
        #     mlflow.log_metric("Recall-Score", recall_score)
        #     mlflow.log_metric("Accuracy-Score", accuracy_score)

        #     if tracking_url_type_store != "file":
        #         # Register the model with its name
        #         log_model(best_model, "model", registered_model_name=model_name)
        #     else:
        #         log_model(best_model, "model")

    def train_model(self,X_train,y_train,X_test,y_test):
        try:

            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
        #     params={
        #     "Decision Tree": {
        #         'criterion':['gini', 'entropy', 'log_loss'],
        #         # 'splitter':['best','random'],
        #         # 'max_features':['sqrt','log2'],
        #     },
        #     "Random Forest":{
        #         # 'criterion':['gini', 'entropy', 'log_loss'],
                
        #         # 'max_features':['sqrt','log2',None],
        #         'n_estimators': [8,16,32,128]
        #     },
        #     "Gradient Boosting":{
        #         # 'loss':['log_loss', 'exponential'],
        #         'learning_rate':[.1,.01,.05,.001],
        #         'subsample':[0.6,0.75,0.85,0.9],
        #         # 'criterion':['squared_error', 'friedman_mse'],
        #         # 'max_features':['auto','sqrt','log2'],
        #         'n_estimators': [8,16,32,64,128,256]
        #     },
        #     "Logistic Regression":{},
        #     "AdaBoost":{
        #         'learning_rate':[.1,.01,.001],
        #         'n_estimators': [8,16,32,64,128]
        #     }
            
        # }
            # model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)
            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score = max(list(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            classification_train_metrics = get_classification_score(y_true=y_train,y_pred=y_train_pred)
            classification_test_metrics = get_classification_score(y_true=y_test,y_pred=y_test_pred)

            self.track_mlflow(best_model, classification_train_metrics, best_model_name)
            self.track_mlflow(best_model, classification_test_metrics, best_model_name)

            logging.info("self.trackmlflow chl gya")


            preprocessor = load_object(file_path=self.data_transformation_artifacts.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)
            Network_Model=NetworkModel(preprocessor=preprocessor,model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path,obj=Network_Model)
            model_trainer_artifact=ModelTrainerArtifacts(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metrics,
                             test_metric_artifact=classification_test_metrics
            
            )
            save_object("final_model/model.pkl",best_model)                 
            
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")   
            return model_trainer_artifact
        except Exception as e:    
            raise NetworkSecurityException(e,sys)
    
    def initaite_model_trainer(self)->ModelTrainerArtifacts:
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
            
            logging.info("train test spiit hogya")

            model_trainer_artifact = self.train_model(X_train,y_train,X_test,y_test) 
            return model_trainer_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)


