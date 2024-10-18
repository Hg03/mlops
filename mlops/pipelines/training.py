from comet_ml import Experiment, API
from mlops.my_hopworks.connector import FeatureStoreManager
from config import load_config
from dotenv import load_dotenv
import polars as pl
from xgboost import XGBClassifier
import joblib
import secrets
import string
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
import os

class training_pipeline:
    def __init__(self, current_time, path):
        self.configs = load_config()
        self.current_time = current_time
        self.path = path
    
    def generate_api_key(self, length: int = 32) -> str:
        alphabet = string.ascii_letters + string.digits
        return "".join(secrets.choice(alphabet) for _ in range(length))
    
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
        exp_key = self.generate_api_key()
        experiment = Experiment(api_key=comet_ml_api_key, project_name="purchase-prediction", workspace='harish-workspace', experiment_key=exp_key)
        X_train = self.train.drop("purchasestatus")
        y_train = self.train["purchasestatus"]
        X_test = self.test.drop("purchasestatus")
        y_test = self.test["purchasestatus"]
        params = self.configs.training.xgboost
        experiment.log_parameters(params)
        model = XGBClassifier(**params)
        
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

        scores = {
            "accuracy": accuracy_score(y_test_series, y_pred_series),
            "precision": precision_score(y_test_series, y_pred_series),
            "recall": recall_score(y_test_series, y_pred_series),
            "roc_auc": roc_auc_score(y_test_series, y_pred_series)
            }
                
        experiment.log_metrics(scores)
        experiment.log_confusion_matrix(y_test_series, y_pred_series)
        precision, recall, _ = precision_recall_curve(y_test_series, y_pred_series)

        # Log the curve to Comet
        experiment.log_curve(
            name="precision-recall",
            x=precision,
            y=recall
        )
        # Add prediction and ground truth columns to the test DataFrame
        debug_df = X_test.with_columns([y_pred_series, y_test_series])
        # Log the DataFrame as a CSV to Comet.ml
        experiment.log_table("prediction_debug_table.csv", debug_df.to_pandas())
        joblib.dump(model, os.path.join(self.path["model"], self.configs.path.model_file))
        experiment.log_model("model", os.path.join(self.path["model"], self.configs.path.model_file))
        # experiment.register_model("model", version='1')
        # Register the model in the model registry
        api = API(api_key=comet_ml_api_key)
        
        api_experiment = api.get(f'harish-workspace/purchase-prediction/{exp_key}')
        api_experiment.register_model("model")
        #     model_name='model',
        #     version='1.0.0',
        #     comment='Initial model registration',
        #     file_name=os.path.join(self.path["model"], self.configs.path.model_file)
        #     )
        print('logged !!')


    def run(self):
        train, test = self.load_from_hopsworks()
        print("success loading")
        self.tune_and_train()
        

if __name__ == "__main__":
    pass

        
