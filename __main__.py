from config import load_config
from mlops.pipelines.preprocess import feature_pipeline
from mlops.pipelines.training import training_pipeline
from datetime import datetime
import os

class purchase_prediction:
    def __init__(self, current_time):
        self.configs = load_config()
        self.current_time = current_time
        self.make_dir(self.current_time)
        
    def make_dir(self, current_time):
        os.makedirs(self.configs.path.output, exist_ok=True)
        self.output_path = os.path.join(self.configs.path.output, current_time)
        os.makedirs(self.output_path, exist_ok=True)
        self.split_path = os.path.join(self.output_path, self.configs.path.split)
        os.makedirs(self.split_path, exist_ok=True)
        self.preprocessed_path = os.path.join(self.output_path, self.configs.path.preprocessed)
        os.makedirs(self.preprocessed_path, exist_ok=True)
    
    def execute_run(self):
        feature_pipeline_obj = feature_pipeline(self.current_time)
        # feature_pipeline_obj.run()
        training_pipeline_obj = training_pipeline(self.current_time)
        training_pipeline_obj.run()
        

if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M") + '/'
    instance = purchase_prediction(current_time=current_time)
    instance.execute_run()
