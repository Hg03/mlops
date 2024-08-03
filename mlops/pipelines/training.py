from mlops.my_hopworks.connector import FeatureStoreManager
from config import load_config
from dotenv import load_dotenv
from comet_ml import Experiment
import polars as pl
from xgboost import XGBClassifier
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
        comet_ml_api_key = os.getenv("comet_ml_api_key")
        experiment = Experiment(api_key=comet_ml_api_key, project_name="purchase-prediction")
        X_train, y_train = self.train.drop("purchasestatus"), self.train.select("purchasestatus")
        X_test, y_test = self.test.drop("purchasestatus"), self.test.select("purchasestatus")
        model = XGBClassifier(n_estimators=10, max_depth=5, learning_rate=0.1, objective="binary:logistic")
        # fit model
        model.fit(X_train,y_train,eval_set=[(X_test, y_test)])
        # make predictions
        y_pred = model.predict(X_test)
        debug_df = pl.DataFrame(X_test)

        debug_df["pred"] = y_pred
        debug_df["ground_truth"] = y_test

        experiment.log_table("prediction_debug_table.csv", debug_df)
        print('logged !!')

        
        
        
    def run(self):
        train, test = self.load_from_hopsworks()
        print("success loading")
        self.tune_and_train()
        

if __name__ == "__main__":
    pass

        
