from config import load_config
from dotenv import load_dotenv
from supabase import create_client
from sklearn import model_selection, compose, pipeline, preprocessing, impute
import polars as pl
from datetime import datetime
import hopsworks
import os


class FeatureStoreManager:
    def __init__(self, api_key_value):
        self.project = hopsworks.login(api_key_value=api_key_value)
        self.fs = self.project.get_feature_store()
        self.mr = self.project.get_model_registry()

    def get_feature_group(self, feature_group_name, version=1):
        return self.fs.get_feature_group(feature_group_name, version=version)

    def get_or_create_feature_group(self, feature_group_name, version=1, description=None, primary_key=None):
        return self.fs.get_or_create_feature_group(name=feature_group_name, version=version, description=description, primary_key=primary_key)

    def insert_data_into_feature_group(self, feature_group, data_frame, write_options=None):
        feature_group.insert(data_frame, write_options=write_options)
        print('Insert Done')

class feature_pipeline:
    def __init__(self):
        load_dotenv()
        self.url = os.getenv("supabase_url")
        self.key = os.getenv("supabase_key")
        self.configs = load_config()
        self.make_dir()
    
    def make_dir(self):
        self.current_time =  datetime.now().strftime("%Y-%m-%d-%H-%M") + '/'
        os.makedirs(self.configs.path.output, exist_ok=True)
        self.output_path = os.path.join(self.configs.path.output, self.current_time)
        os.makedirs(self.output_path, exist_ok=True)
        self.split_path = os.path.join(self.output_path, self.configs.path.split)
        os.makedirs(self.split_path, exist_ok=True)
        self.preprocessed_path = os.path.join(self.output_path, self.configs.path.preprocessed)
        os.makedirs(self.preprocessed_path, exist_ok=True)
        
    def loader(self):
        conn = create_client(self.url, self.key)
        json_data = []
        batch_size = self.configs.preprocess.batch_size
        offset = self.configs.preprocess.offset
        while True:
            response = conn.table(self.configs.path.table).select("*").limit(batch_size).offset(offset).execute()
            batch = response.data
            if not batch:
                break
            json_data.extend(batch)
            offset += batch_size
        self.raw_data = pl.DataFrame(json_data)
        self.raw_data = self.raw_data.drop(["created_at"])
        self.raw_data.write_parquet(os.path.join(self.output_path, self.configs.path.raw))
        return self.raw_data
    
    def splitter(self):
        self.train, self.test = model_selection.train_test_split(self.raw_data, test_size=self.configs.preprocess.test_size, shuffle=True)
        self.train.write_parquet(os.path.join(self.split_path, self.configs.path.train_file))
        self.test.write_parquet(os.path.join(self.split_path, self.configs.path.test_file))

        return self.train, self.test
    
    def get_columns_preserver(self, X):
        original_columns = list(X.columns)
        updated_columns = [col[col.rfind('__') + 2:] for col in original_columns]
        X = X.rename(columns=dict(zip(original_columns, updated_columns)))
        return X
    
    def get_preprocessing_pipeline(self, column_configs, switches, strategies):
        steps = []

        # Define feature preserver step
        feature_preserver = preprocessing.FunctionTransformer(self.get_columns_preserver)

        # Imputation step
        if switches.get("impute"):
            imputation_transformers = []
            if column_configs.get("numeric"):
                imputation_transformers.append(
                    ("numerical_imputer", impute.SimpleImputer(strategy=strategies.get("impute_numeric", "mean")), column_configs.numeric)
                )
            if column_configs.get("categoric"):
                cat_cols = column_configs.categoric.ordinal + column_configs.categoric.nominal
                imputation_transformers.append(
                    ("categorical_imputer", impute.SimpleImputer(strategy=strategies.get("impute_categoric", "most_frequent")), cat_cols)
                )
        
            imputation_pipeline = compose.ColumnTransformer(
                transformers=imputation_transformers,
                remainder='passthrough'
            ).set_output(transform='polars')
        
            steps.append(('imputer', pipeline.Pipeline([
                ('imputation', imputation_pipeline),
                ('feature_preserver', feature_preserver)
            ])))
        
        # Encoding step
        if switches.get("encode"):
            encoding_transformers = []
            if column_configs.get("categoric"):
                if column_configs["categoric"].get("ordinal"):
                    encoding_transformers.append(
                        ("ordinal_encoder", preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), column_configs.categoric.ordinal)
                    )
                if column_configs["categoric"].get("nominal"):
                    encoding_transformers.append(
                        ("nominal_encoder", preprocessing.OneHotEncoder(sparse_output=False, handle_unknown='ignore'), column_configs.categoric.nominal)
                    )
            
            encoding_pipeline = compose.ColumnTransformer(
                transformers=encoding_transformers,
                remainder='passthrough'
            ).set_output(transform='polars')
            
            steps.append(('encoder', pipeline.Pipeline([
                ('encoding', encoding_pipeline),
                ('feature_preserver', feature_preserver)
            ])))

        self.preprocessing_pipeline = pipeline.Pipeline(steps=steps)
        # Create final pipeline
        return self.preprocessing_pipeline
    
    def prep_the_data(self):
        train = pl.read_parquet(os.path.join(self.split_path, self.configs.path.train_file))
        test = pl.read_parquet(os.path.join(self.split_path, self.configs.path.test_file))
        # self.preprocessed_train = self.preprocessing_pipeline.fit_transform(train)
        # self.preprocessed_test = self.preprocessing_pipeline.transform(test)
        # self.preprocessed_train.write_parquet(os.path.join(self.preprocessed_path, self.configs.path.train_file))
        # self.preprocessed_test.write_parquet(os.path.join(self.preprocessed_path, self.configs.path.test_file))
        # return self.preprocessed_train, self.preprocessed_test
        return train, test
    
    def load_to_hopsworks(self):
        load_dotenv()
        api_key_value = os.getenv("hopsworks_api_key")
        manager = FeatureStoreManager(api_key_value=api_key_value)
        train_fg = manager.get_or_create_feature_group(
                        feature_group_name="training",
                        version=1,
                        description="training data",
                        primary_key=["id"]
                    )
        test_fg = manager.get_or_create_feature_group(
            feature_group_name="testing",
            version=1,
            description="testing data",
            primary_key=["id"]
        )
        write_options = {"wait_for_job": True}
        manager.insert_data_into_feature_group(feature_group=train_fg, data_frame=self.train.to_pandas(), write_options=write_options)
        manager.insert_data_into_feature_group(feature_group=test_fg, data_frame=self.test.to_pandas(), write_options=write_options)
        print("Data inserted to feature store successfully !!")
    
    def run(self):
        self.raw_data = self.loader()
        self.train, self.test = self.splitter()
        self.preprocessing_pipeline = self.get_preprocessing_pipeline(self.configs.columns, self.configs.preprocess.switches, self.configs.preprocess.strategies)
        self.preprocessed_train, self.preprocessed_test = self.prep_the_data()
        self.load_to_hopsworks()
        
if __name__ == "__main__":
    inst = feature_pipeline()
    inst.run()
        
        
        