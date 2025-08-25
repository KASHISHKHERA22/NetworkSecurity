from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifacts_entity import ClassificationMetricArtifact

import os   
import sys
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def get_classification_score(y_true,y_pred)->ClassificationMetricArtifact:
    try:
        model_f1_score = f1_score(y_true,y_pred,average="weighted")
        model_precision_score = precision_score(y_true,y_pred,average="weighted")
        model_recall_score = recall_score(y_true,y_pred,average="weighted")
        model_accuracy_score = accuracy_score(y_true,y_pred)
        classification_metric_artifact = ClassificationMetricArtifact(f1_score=float(model_f1_score),precision_score=float(model_precision_score),recall_score=float(model_recall_score),accuracy_score=float(model_accuracy_score))
        return classification_metric_artifact

    except Exception as e:
        raise NetworkSecurityException(e, sys)  