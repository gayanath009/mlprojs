import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score, accuracy_score

def save_object(obj, file_path):
      
    try:
        dir_path = os.path.dirname(file_path) # Get the directory path of the file
        os.makedirs(dir_path, exist_ok=True) # Create the directory if it doesn't exist
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)
    

# def evaluate_models(models, X_train, y_train, X_test, y_test):
#         try:
#             report ={}
#             for i in range(len(list(models))):

#                 models.fit(X_train, y_train)  # Fit the model on the training data
#                 y_train_pred = models.predict(X_train)  # Predict using the model
#                 y_test_pred = models.predict(X_test)
#                 train_model_score = r2_score(y_train, y_train_pred)  # Calculate training accuracy
#                 test_model_score = r2_score(y_test, y_test_pred)

#                 report[str(models)] = train_model_score
                
#             return report
#         except Exception as e:
#             raise CustomException(e, sys)

def evaluate_models(models, X_train, y_train, X_test, y_test):
    try:
        report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)  # Fit the model
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Use appropriate metric depending on model type (regressor or classifier)
            try:
                score = r2_score(y_test, y_test_pred)  # For regression
            except:
                score = accuracy_score(y_test, y_test_pred)  # For classification

            report[name] = score
        return report
    except Exception as e:
        raise CustomException(e, sys)
