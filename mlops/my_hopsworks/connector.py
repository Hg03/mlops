import hopsworks

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
