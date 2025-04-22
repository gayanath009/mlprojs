import os
import sys
from dataclasses import dataclass
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
) 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models  

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")



class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")    
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Linear Regression": LinearRegression(),
                "KNeighbors": KNeighborsClassifier(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostClassifier(),
                "KNeighbors": KNeighborsClassifier(),
                "AdaBoost": AdaBoostClassifier(),
            }

            model_report = evaluate_models(
               X_train = X_train,  y_train = y_train,  X_test = X_test, y_test = y_test, models = models
            )

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            best_model = models[best_model_name]

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )


            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"R2 score of the best model: {r2_square}")
            return r2_square, best_model_name, best_model_score
        

        except Exception as e:
            raise CustomException(e, sys) from e

