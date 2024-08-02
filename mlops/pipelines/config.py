from omegaconf import OmegaConf

def load_config():
    path_config = OmegaConf.load('conf/path.yaml')
    columns_config = OmegaConf.load('conf/columns.yaml')
    preprocess_config = OmegaConf.load('conf/preprocess.yaml')
    training_config = OmegaConf.load('conf/training.yaml')
    inference_config = OmegaConf.load('conf/inference.yaml')
    merged_config = OmegaConf.merge(path_config, columns_config, preprocess_config, training_config, inference_config)
    configs = OmegaConf.to_yaml(merged_config)
    return merged_config

if __name__ == "__main__":
    configs = load_config()
    print(configs.columns)
    