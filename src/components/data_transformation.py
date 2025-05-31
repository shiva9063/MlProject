import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_object(self):
        ''' this function is resiponsible for the data transformation'''
        try:
            numerical_columns=['reading_score', 'writing_score']
            categorical_columns=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            num_pipline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                     ('scaler',StandardScaler())
                     ]
            )
            cat_pipline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoding',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))

                ]
            )
            logging.info('standard scaling completed')
            logging.info('categorical encoding completed')
            preprocessor=ColumnTransformer([
                ('num_pipline',num_pipline,numerical_columns),
                ('cat_pipline',cat_pipline,categorical_columns)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('read train and test data completed')
            logging.info('obtaining preprocessing object')
            preprocessing_obj=self.get_data_transformer_object()
            target_column_name='math_score'
            numerical_columns=['math_score', 'reading_score', 'writing_score']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            traget_feature_test_df=test_df[target_column_name]

            logging.info(f'applying preproceesing on training adn testing dat set')

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_arr,np.array(traget_feature_test_df)
            ]
            logging.info(f'saving preprocessing object')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,

            )
        except Exception as e:
            raise CustomException(e,sys)