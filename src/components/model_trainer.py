import os 
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor ,AdaBoostRegressor,GradientBoostingRegressor)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor   
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models
## code

@dataclass
class ModelTrainerConfig:
    trained_model_path=os.path.join('artifacts','model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('spliting training and test input data')
            X_train,Y_train, X_test, Y_test = (train_array[:, :-1],train_array[:, -1],test_array[:, :-1],test_array[:, -1])

            models={
                'Random Forest':RandomForestRegressor(),
                'Decision Tree':DecisionTreeRegressor(),
                'Gradiant Boosting':GradientBoostingRegressor(),
                'AdaBoost':AdaBoostRegressor(),
                'Linear Regression':LinearRegression(),
                'CatBoost':CatBoostRegressor(),
                'XGBRegressor':XGBRegressor(),
                'KNeighborsRegressor':KNeighborsRegressor(),
            }
            model_report:dict=evaluate_models(X_train=X_train,Y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models)
            ## to get best model score from dict
            best_model_score=max(sorted(model_report.values()))
            ## best model name form dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException('No Best model Found')
            logging.info('best model found on the traiing and testing dataset')
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2=r2_score(predicted,Y_test)
            return r2
        except Exception as e:
            raise CustomException(e,sys)
            