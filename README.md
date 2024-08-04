# Purchase Prediction

![image](https://github.com/user-attachments/assets/9c37cf95-bd94-44dd-981a-09693d296df6)

## Introduction
This represents a full stack mlops projects predicting that customer will purchase the product or not based on data provided. Get some glimplse of dataset [here](https://www.kaggle.com/datasets/rabieelkharoua/predict-customer-purchase-behavior-dataset).

## Approach
As a basic approach, we are about to create mlops infra in 3 pipeline structure i.e.

- Feature Pipeline (I named it preprocess): Here we'll be loading the data from **supabase** and treating it as a polars dataframe, preprocessing the data using sklearn's features and exporting the preprocessed data to feature store named **hopsworks**.
- Training Pipeline: Here, we'll load the preprocessed data from **hopsworks**, perform the hyperparameter tuning of model and create the model with best params and train the model on the loaded data. All the parameters, metrics, curve and model will be tracked to **comet ml**.
- Inference Pipeline: Not reached here.

## Resource
- [supabase](supabase.com)
- [comet ml](comet.com)
- [hopsworks](hopsworks.ai)
- [polars](pola.rs)
