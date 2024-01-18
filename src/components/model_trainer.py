# Basic Import
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
from dataclasses import dataclass

from src.utils import save_object,evaluate_models
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting traning and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Logistic regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "K-Neighbors": KNeighborsClassifier(),
                "XGBClassifier": xgb.XGBClassifier(),
                "Cat Boosting": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier()
            }
            '''
            This are the parameters of hyperparameter tuning.
            Not doing this because its taking lot of time due to low specifications of device.
            and getting good precision, accuracy and recall witout doing hyperparameter tuning.


            params={
                "Decision Tree": {
                    'criterion':['entropy', 'gini'],
                    'max_depth': [5, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]

                },
                "Random Forest":{
                    'n_estimators': [50,100,200],
                    'max_depth': [5, 10, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]            

                },

                "K-Neighbors": {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "Linear Regression":{},

                "XGBRegressor":{
                    'learning_rate': [0.1, 0.01, 0.05],
                    'subsample': [0.6, 0.7, 0.8, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]


                },
                "CatBoosting Regressor":{
                    'depth': [4,6,8],
                    'learning_rate': [0.01, 0.2, 0.1],
                    'iterations': [50, 100, 200]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[0.01,0.1,0.2],
                    'n_estimators': [50,100,200]
                },
                "Logistic regression": {}
                
            }
            '''

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            
            ## To get best model score from dict
            best_model_score= max(sorted(model_report.values()))

            ## To get best model name from dict
            best_model_name= list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model= models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found on both traning and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted= best_model.predict(X_test)
            confusion= confusion_matrix(y_test, predicted)
            accuracy= accuracy_score(y_test, predicted)
            precision= precision_score(y_test, predicted)
            recall= recall_score(y_test, predicted)


            return confusion, accuracy, precision, recall
                     
        

        except Exception as e:
            raise CustomException(e,sys)

            
