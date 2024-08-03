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

        # Assuming self.train and self.test are Polars DataFrames
        X_train = self.train.drop("purchasestatus")
        y_train = self.train["purchasestatus"]
        X_test = self.test.drop("purchasestatus")
        y_test = self.test["purchasestatus"]

        model = XGBClassifier(n_estimators=10, max_depth=5, learning_rate=0.1, objective="binary:logistic")
        
        # Convert Polars DataFrame to numpy for training
        X_train_np = X_train.to_numpy()
        y_train_np = y_train.to_numpy().ravel()  # Ensure y is a 1D array
        X_test_np = X_test.to_numpy()
        y_test_np = y_test.to_numpy().ravel()  # Ensure y is a 1D array

        # Fit model
        model.fit(X_train_np, y_train_np, eval_set=[(X_test_np, y_test_np)])
        
        # Make predictions
        y_pred = model.predict(X_test_np)

        # Convert numpy array to Polars Series
        y_pred_series = pl.Series("pred", y_pred)
        y_test_series = pl.Series("ground_truth", y_test_np)

        # Add prediction and ground truth columns to the test DataFrame
        debug_df = X_test.with_columns([y_pred_series, y_test_series])

        # Log the DataFrame as a CSV to Comet.ml
        experiment.log_table("prediction_debug_table.csv", debug_df.to_pandas())
        print('logged !!')

        
        
        
    def run(self):
        train, test = self.load_from_hopsworks()
        print("success loading")
        self.tune_and_train()
        

if __name__ == "__main__":
    pass

        
