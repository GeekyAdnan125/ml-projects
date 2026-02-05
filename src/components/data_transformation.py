import pandas as pd 
import numpy as np 
import sys 
import os 
from src.utils import save_object
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

@dataclass
class DataTransformationConfig:
    preprocesser_obj_file_path = os.path.join('artifacts','preprocesser.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        this function is responsible for data transformation 
        """
        try:
            logging.info("in get_data_transformer_object")
            numerical_cols = ['writing_score' , 'reading_score']
            categorical_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            ## now create a pipline 
            num_pipeline = Pipeline(
                steps=[
                    ("imputer" , SimpleImputer(strategy="median")),
                    ("scaler" , StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline  = Pipeline(
                steps=[
                    ("inputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder" , OneHotEncoder(handle_unknown="ignore")),
                    ("scaler" , StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline" , num_pipeline,numerical_cols),
                    ("cat_pipeline" , cat_pipeline,categorical_cols)
                ]
            )
            logging.info("Numerical columns scaling completed")
            logging.info("categorical columns encoding completed")
            return preprocessor
        except Exception as e :
            raise CustomException(e,sys)
        


    def initate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_cols = ['writing_score' , 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_traning_df =train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df =test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")  

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df) 
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)  

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_traning_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info(f"saved preprocessed object")

            save_object(
                file_path = self.data_transformation_config.preprocesser_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocesser_obj_file_path
            )
        except Exception as e :
            raise Exception(e,sys)

        
