from dotenv import load_dotenv
import os
import sys
import pandas as pd
import numpy as np
import json
import pymongo
import certifi
ca = certifi.where()
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)


class NetworkDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def csv_to_json(self,file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    # def insert_data_to_mongodb(self,database,collection,records):
    #     try:
    #         self.database = database
    #         self.collection = collection
    #         self.records = records
    #         self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
    #         self.database = self.mongo_client[self.database]
    #         self.collection = self.database[self.collection]
    #         self.collection.insert_many(self.records)    
    #     except Exception as e:
    #         raise NetworkSecurityException(e, sys)  

    def insert_data_to_mongodb(self, database, collection, records):
        try:
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            db = self.mongo_client[database]
            coll = db[collection]
            result = coll.insert_many(records)
            return len(result.inserted_ids)
        except Exception as e:
            raise NetworkSecurityException(e, sys)  
    
if __name__ == "__main__":
    FILE_PATH = "Network_Data/phisingData.csv"
    DATABASE = "KASHISH"
    COLLECTION = "NETWORK_DATA"
    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json(file_path=FILE_PATH)
    no_of_records=networkobj.insert_data_to_mongodb(DATABASE,COLLECTION,records)
    print(no_of_records)