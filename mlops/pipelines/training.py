from mlops.my_hopworks.connector import FeatureStoreManager
from config import load_config
from dotenv import load_dotenv
import polars as pl
import os

class training_pipeline:
    def __init__(self, current_time):
        self.configs = load_config()
        self.current_time = current_time
    
    def load_from_hopsworks(self):
        load_dotenv()
        hopsworks_api_key_value: str = os.getenv("hopsworks_api_key")
        manager = FeatureStoreManager(hopsworks_api_key_value)
        train_fg = manager.get_feature_group("training", version=1)
        test_fg = manager.get_feature_group("testing", version=1)
        query_train = train_fg.select_except(["id"])
        query_test = test_fg.select_except(["id"])
        self.train = pl.DataFrame(query_train.read())
        self.test = pl.DataFrame(query_test.read())
        return self.train, self.test
    
    def tune_and_train(self):
        ...
        
    def run(self):
        train, test = self.load_from_hopsworks()
        print("success loading")
        self.tune_and_train()
        

if __name__ == "__main__":
    pass

        
