import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
from src.exception import CustomeException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('Artificat', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_obj(self):
        try:
            numerical_column = ["writing score", "reading score"]
            categotical_column = ["gender", 
                                "race/ethnicity",                 
                                "parental level of education",    
                                "lunch",                           
                                "test preparation course"]
            
            num_pipeline = Pipeline(
                
                steps = [
                    ("Imputer", SimpleImputer(strategy = "median")),
                    ("Scaler", StandardScaler(with_mean = False)),
                     ])
                    
            cat_pipeline = Pipeline(
                steps = [
                    ("Imputer",SimpleImputer(strategy = "most_frequent")),
                    ("ONH", OneHotEncoder()),
                    ("Scaler", StandardScaler(with_mean = False)),
                    ]
            ) 
            logging.info("Numerical columns standard scaling")      
            logging.info("Categorical columns are encoded")
            
            preprocessor = ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, numerical_column),
                ("cat_pipline", cat_pipeline, categotical_column)
                    
                ]
                
            )
            return preprocessor
        except Exception as e:
            raise CustomeException(e,sys)
            
    def initiate_data_transformation(self, train_path, test_path):  
         try:
             train_df = pd.read_csv(train_path)
             test_df = pd.read_csv(test_path)
             logging.info("Reading the train and test data")
             logging.info("Obtaining preprocessing object")
             
             preprocessing_obj = self.get_data_transformer_obj()
             target_column = "math score"
             
             input_feat_train_df = train_df.drop(columns = [target_column], axis=1)
             target_feat_train_df = train_df[target_column]
             
             input_feat_test_df = test_df.drop(columns = [target_column], axis=1)
             target_feat_test_df = test_df[target_column]
             
             
             logging.info(f"Applying preprocessing object on training and test dataframe.")
             
             input_feature_train_arr = preprocessing_obj.fit_transform(input_feat_train_df)
             input_feature_test_arr = preprocessing_obj.transform(input_feat_test_df)
             
             train_arr = np.c_[
                 input_feature_train_arr, np.array(target_feat_train_df)
             ]
             test_arr = np.c_[input_feature_test_arr, np.array(target_feat_test_df)]
             
             logging.info(f"Saved preprocessing object.")
             
             save_object(
                 file_path = self.data_transformation_config.preprocessor_obj_file_path,
                 obj = preprocessing_obj
             )
             return(
                 train_arr,
                 test_arr,
                 self.data_transformation_config.preprocessor_obj_file_path
             )
         except Exception as e:
             raise CustomeException(e, sys)   
    

