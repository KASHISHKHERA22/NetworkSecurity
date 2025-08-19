from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os
import dill
import sys
import yaml
import pickle

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its content as a dictionary. 
    """
    try:
        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
            raise NetworkSecurityException(e, sys)  